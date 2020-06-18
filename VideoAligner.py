from collections import Counter
from logging import LoggerAdapter, getLogger
from multiprocessing import cpu_count
import time
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

import cv2
from joblib import Parallel, delayed
import numpy as np
from scipy.interpolate import interp1d
from skimage.transform import EuclideanTransform
from skimage.measure import ransac


class VideoAligner:
    class AlignmentError(BaseException):
        pass

    FRAME_SAMPLE_RATE = 100  # Hz
    SPATIAL_DOWNSAMPLE_RATE = 1
    N_JOBS_PARALLEL = cpu_count()
    DETECTOR_CONSTRUCTOR_DICT = {
        "akaze": cv2.AKAZE_create,
        "brisk": cv2.BRISK_create,
    }
    # threshold at which exception is raised
    N_KP_GLOBAL_MIN = 5
    # threshold at which frame transformation estimate is skipped and interpolated instead
    N_KP_FRAME_SKIP = 3
    # fractional number of total frames prior to template frame
    TEMPLATE_FRAME_LOC = 0.5
    # max fraction of interpolated frames before error is raised
    MAX_FRAC_INTERPOLATED = 0.2
    # maximum ratio of of closest to second closest keypoint descriptor-space distance
    DESCRIPTOR_DISTANCE_RATIO_THRESH = 0.75
    # range based on fraction of median distance between matched keypoints for match to be inlier
    MEDIAN_KEYPOINT_INLIER_DISTANCE_RANGE = (0.5, 2.0)
    # intensity percentile to be used as maximum for max-scaling normalization of video
    IMAGE_NORM_MAX_PERCENTILE = 99.99
    MAX_PIXEL_UINT8 = 255
    # RANSAC parameters
    RANSAC_MIN_SAMPLES = 2
    RANSAC_RESIDUAL_THRESH = 2
    RANSAC_MAX_TRIALS = 1000
    RANDOM_SEED = 42

    def __init__(self, logger: LoggerAdapter = None):
        self.logger = logger
        if self.logger is None:
            self.logger = getLogger(self.__class__.__name__)
        self.affines = None
        self.aligned_imgs = None
        self.interpolated_idxs = None
        self._kp_template = None
        self._des_template = None

    def align_images(
        self,
        images: np.ndarray,
        n_kp_global: int,
        detector_algorithm: str,
        frame_rate: int,
        masked_template: Optional[np.ndarray] = None,
        patch: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Registers and aligns a set of frames from a  video to a cropped_template.
        The alignment can use any algorithm from `VideoAligner.DETECTOR_CONSTRUCTOR_DICT` to identify keypoints
        for each frame, which are matched to the those from the template frame. After the initial
        matching, the matches are filtered to the `n_kp` which received the most matches between the
        template and video frames. For each frame, the optimal euclidean transform is calculated via
        RANSAC to align the filtered keypoints between the current frame and the template frame.
        Args:
            images: [frames, x, y] stack of images, usually from `Movie.raw_matrix`
            n_kp_global: number of highest voted keypoints to keep for transformation estimation
            detector_algorithm: lowercase detector name from keys of `VideoAligner.DETECTOR_CONSTRUCTOR_DICT`
            masked_template: [x, y] a template frame which has been masked by
            frame_rate: framerate of movie used to calculate `frame_downsample_rate` based on
                `FRAME_SAMPLE_RATE`
            patch: changes return to a spatial subset defined by (x, y, width, height), where the
                patch starts at (x, y) and ends at (x + width, y + height)

        Returns:
            aligned_imgs: [frames, x, y] images after transformation
            euclidean_transforms: [n_frames, 3] representation of Euclidean transforms in the
                format: [x_translation, y_translation, rotation (radians)]
            skipped_idxs: indices of frames for which affine transforms were not estimated, and
                interpolated instead
        """
        self.logger.info(f"aligning {len(images)} frames with n_kp_global: {n_kp_global} and {detector_algorithm}")
        if not 0 <= self.TEMPLATE_FRAME_LOC <= 1:
            raise ValueError("`template_frame_loc` must be between 0 and 1")
        if masked_template is not None:
            template = masked_template  # use the auto-cropped template frame
        else:
            template_idx = int(len(images) * self.TEMPLATE_FRAME_LOC)
            template = images[template_idx]
            self.logger.info(f"no masked template provided. Using frame {template_idx} as template")

        self.logger.info("normalizing frames...")
        t_start = time.time()
        brightest_px = self._get_brightest_px(images)
        images_i8, template_i8 = self._max_scale_images(images, template, brightest_px, np.uint8)
        assert images.dtype == np.uint16 and images_i8.dtype == np.uint8
        frame_downsample_rate = max(1, frame_rate // self.FRAME_SAMPLE_RATE)
        images_sample, template = self._downsample(
            images_i8, template_i8, frame_downsample_rate, self.SPATIAL_DOWNSAMPLE_RATE
        )
        self.logger.info(f"normalized frames in: {round(time.time() - t_start)} s")

        self.logger.info("identifying keypoints...")
        t_start = time.time()
        n_frames_sample = images_sample.shape[0]
        detector = self.DETECTOR_CONSTRUCTOR_DICT[detector_algorithm]()
        self._kp_template, self._des_template = detector.detectAndCompute(template, None)
        self._kp_template = np.array([p.pt for p in self._kp_template])
        matched_data = self._parallelize_i(
            self._get_frame_keypoints,
            images_sample,
            kp_template=self._kp_template,
            des_template=self._des_template,
            detector_algorithm=detector_algorithm,
        )
        (kp_idxs_list, kp_query_list, log_strs) = zip(*matched_data)
        for str_ in log_strs:
            self.logger.debug(str_)
        self.logger.info(f"identified keypoints in: {round(time.time() - t_start)} s")

        self.logger.info("generating keypoint consensus...")
        t_start = time.time()
        consensus_idxs = self._get_consensus_kps(kp_idxs_list, n_frames_sample, n_kp_global)
        frames_kp_template, frames_kp_query = self._lookup_consensus_kps(consensus_idxs, kp_idxs_list, kp_query_list)
        self.logger.info(f"generated keypoint consensus in: {round(time.time() - t_start)} s")

        self.logger.info("estimating affine transformations...")
        t_start = time.time()
        affines_sample = self._parallelize(
            self._compute_euclidean_affine,
            frames_kp_template,
            frames_kp_query,
            spatial_downsample_rate=self.SPATIAL_DOWNSAMPLE_RATE,
        )
        affines, skipped_idxs = self._process_affines(affines_sample, frame_downsample_rate)
        affines, self.interpolated_idxs = self._interpolate_affines(affines)
        euclidean_transforms = self._get_euclidean_transforms(affines)
        self.logger.info(f"generated keypoint consensus in: {round(time.time() - t_start)} s")

        self.logger.info("transforming frames...")
        t_start = time.time()
        aligned_imgs = self._parallelize(self._apply_affine, images, affines)
        aligned_imgs = np.stack(aligned_imgs)
        self.logger.info(f"transformed frames in: {round(time.time() - t_start)} s")
        if patch is not None:
            x_start, y_start, width, height = patch
            x_end = x_start + width
            y_end = y_start + height
            aligned_imgs = aligned_imgs[:, x_start:x_end, y_start:y_end]
        return aligned_imgs, euclidean_transforms, skipped_idxs

    @staticmethod
    def _get_frame_keypoints(
        i: int, image: np.ndarray, kp_template: np.ndarray, des_template: np.ndarray, detector_algorithm: str
    ) -> Tuple[Set[int], np.ndarray, str]:
        """
        Gets the frame keypoints, matches them to the template frame, and filters the matches by
        descriptor similarity ratio and distance statistics. The descriptor similarity filter
        removes matches whose best match is not less than `ratio_filter` descriptor-space distance
        compared to the second best match. The distance filter removes matches whose distance is not
        within the `distance_range` of the mean distance between matches. The current frame
        keypoints, `kp_query`, are also reordered to match the template keypoints. The indices of
        the matches are returned, and will be used for each frame to vote for the globally selected
        keypoints.
        Args:
            i: frame number (zero-indexed)
            image:
            kp_template: template keypoints coordinates
            des_template: template keypoint descriptors
            detector_algorithm: one of `VideoAligner.DETECTOR_CONSTRUCTOR_DICT` keys

        Returns:
            kp_idxs: a set which contains integers corresponding to indices of keypoint matches
                between the current frame and the template after filtering. The indices correspond
                to the ordering of the template keypoints
            kp_query: an array where each row contains the (x, y) coordinates of a keypoint from the
                current frame. It is ordered corresponding to the global index (matches with
                template frame)
            log_str: logging output for outer scope (not using logger here because method needed
                to be static for parallelization)
        """
        detector = VideoAligner.DETECTOR_CONSTRUCTOR_DICT[detector_algorithm]()
        kp_query, des_query = detector.detectAndCompute(image, None)
        kp_query = np.array([p.pt for p in kp_query])
        # match features together
        bf = cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(des_template, des_query, k=2)
        # reorder kp_query to match kp_template
        kp_query_ordered = np.zeros_like(kp_template)
        for x in np.array(matches)[:, 0]:
            kp_query_ordered[x.queryIdx] = kp_query[x.trainIdx]
        kp_query = kp_query_ordered
        # ratio filter for feature-space distance between closest and second closest matches
        des_dist_thresh = VideoAligner.DESCRIPTOR_DISTANCE_RATIO_THRESH
        feature_matches = [m for m, n in matches if m.distance < des_dist_thresh * n.distance]
        kp_template_filtered = np.array([kp_template[m.queryIdx] for m in feature_matches])
        kp_query_filtered = np.array([kp_query[m.queryIdx] for m in feature_matches])
        # distance-based outlier rejection
        if feature_matches:
            dists = np.linalg.norm(kp_template_filtered - kp_query_filtered, axis=1)
            (d_1, d_2) = (0.5, 2)
            filter_ = np.where((d_1 * np.median(dists) <= dists) * (dists <= d_2 * np.median(dists)))[0]
            distance_matches = np.array(feature_matches)[filter_]
        else:
            distance_matches = []
        kp_idxs = set([m.queryIdx for m in distance_matches])
        log_str = (
            f"frame {i}:\n"
            f"\t{len(kp_query)} unfiltered features identified\n"
            f"\t{len(matches)} matches identified between frame and template\n"
            f"\t{len(feature_matches)} matches after feature-space ratio filter\n"
            f"\t{len(distance_matches)} matches after distance-based outlier rejection"
        )
        return kp_idxs, kp_query, log_str

    def _get_consensus_kps(self, kp_idxs_list: List[set], n_frames: int, n_kp_global: int) -> Set[int]:
        """
        Identifies the `n_kp` keypoints in the template frame which were matched with keypoints in
        other frames most frequently and returns their global indices.
        Args:
            kp_idxs_list: list of sets, where each set corresponds to a frame, and the elements
                correspond to the global indices of filtered matches between the current frame and
                the template
            n_frames: (logging only) total number of frames in the video for calculating the
                fraction of keypoints in each frame which were voted into the consensus
            n_kp_global: most popular n keypoints to accept

        Returns:
            consensus_idxs: global indices of most popular keypoints amongst frames
        """
        counts = Counter([x for s in kp_idxs_list for x in s])
        votes = [x for x in counts.most_common(n_kp_global)]
        if len(votes) < self.N_KP_GLOBAL_MIN:
            raise VideoAligner.AlignmentError(
                "Too few keypoints found. Try a higher quality video, or decrease " "`VideoAligner.N_KP_GLOBAL_MIN`"
            )
        consensus_idxs, vote_match_rates = zip(*votes)
        vote_match_rates = np.array(vote_match_rates) / n_frames
        self.logger.info(f"top n keypoints match rates: {vote_match_rates}")
        consensus_idxs = set(consensus_idxs)
        return consensus_idxs

    def _lookup_consensus_kps(
        self, consensus_idxs: Set[int], kp_idxs_list: List[Set[int]], kp_query_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        After keypoint voting amongst frames, for each frame, look up the keypoints that are in both
        the current frame and the template frame which match the consensus keypoints.
        Args:
            consensus_idxs: global indices of keypoints that were voted on during consensus
            kp_idxs_list: list of sets, where each set corresponds to a frame, and the elements
                correspond to the global indices of filtered matches between the current frame and
                the template
            kp_query_list: list of arrays, where each array corresponds to a frame, and contains the
                keypoints from that frame

        Returns:
            frames_kp_template: list of arrays, where each array corresponds to the keypoints in the
                template frame which match the current frame and consensus
            frames_kp_query: list of arrays, where each array corresponds to the keypoints in the
                current frame which match the template frame and consensus
        """
        frames_kp_template = []
        frames_kp_query = []
        for i in range(len(kp_query_list)):
            kp_idxs_frame = list(consensus_idxs.intersection(kp_idxs_list[i]))
            kp_template_frame = self._kp_template[kp_idxs_frame]
            kp_query_frame = kp_query_list[i][kp_idxs_frame]
            frames_kp_template.append(kp_template_frame)
            frames_kp_query.append(kp_query_frame)
            if len(kp_query_frame) < self.N_KP_FRAME_SKIP:
                self.logger.info(
                    f"transform for frame {i} not estimated due to low keypoint count: "
                    f"({len(kp_query_frame)}). Will be interpolated based on other frames instead"
                )
        frames_kp_template = np.array(frames_kp_template)
        frames_kp_query = np.array(frames_kp_query)
        return frames_kp_template, frames_kp_query

    @staticmethod
    def _compute_euclidean_affine(
        kp_template: np.ndarray, kp_query: np.ndarray, spatial_downsample_rate: Union[float, int]
    ) -> Optional[np.ndarray]:
        """
        Use the matched keypoints between the a given frame, the template frame, and the consensus
        to estimate the optimal Euclidean transform for alignment via RANSAC. If fewer than `min_kp`
        keypoints are passed, the original image is returned as `aligned_img`, and `None` is
        returned instead of an affine matrix.
        Args:
            kp_template: template keypoints to align
            kp_query: frame keypoints to align

        Returns:
            aligned_img: [x, y] image after alignment. If `ransac` fails silently (returns None), or
                if fewer than `min_kp` keypoints are identified, the
            affine: [2, 3] estimated affine matrix, restricted to euclidean transform
        """
        if len(kp_query) < VideoAligner.N_KP_FRAME_SKIP:
            model = None
        else:
            model, _ = ransac(
                (kp_query, kp_template),
                EuclideanTransform,
                min_samples=VideoAligner.RANSAC_MIN_SAMPLES,
                residual_threshold=VideoAligner.RANSAC_RESIDUAL_THRESH,
                max_trials=VideoAligner.RANSAC_MAX_TRIALS,
                random_state=VideoAligner.RANDOM_SEED,
            )
        # ransac can silently fail and return `None`
        if model is not None:
            affine = model.params[:2]
            affine[:, 2] *= spatial_downsample_rate
        else:
            affine = np.full((2, 3), np.nan)
        return affine

    @staticmethod
    def _process_affines(affines_sample: Sequence[np.ndarray], frame_downsample_rate: int) -> Tuple[np.ndarray, List]:
        """
        Expand affine matrix for interpolation from downsampling and convert list to array
        Args:
            affines_sample: list of affine matrices from sampled frames
            frame_downsample_rate:

        Returns:
            affines: filled affines matrix
            skipped_idxs: indices in sample frames that were skipped. Does not include indices that
                were not estimated due to downsampling
        """
        affines_sample = np.stack(affines_sample)
        skipped_idxs = [i * frame_downsample_rate for i, x in enumerate(affines_sample) if np.isnan(x).any()]
        empty_block_shape = (frame_downsample_rate - 1, 2, 3)
        affines_expanded = [
            np.concatenate([np.expand_dims(x, axis=0), np.full(empty_block_shape, np.nan)]) for x in affines_sample
        ]
        affines = np.concatenate(affines_expanded)
        return affines, skipped_idxs

    @staticmethod
    def _interpolate_affines(affines: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        When an affine matrix cannot be estimated for a particular frame or series of frames, the
        affine matrices for those frames are interpolated or extrapolated and applied to the
        corresponding frames. If the missing transformations are bounded on both sides by estimated
        frames, then they will be linearly interpolated. If they are bounded only on one side, they
        will be filled with the value from the nearest estimated frame.
        Args:

        Returns:

            interpolated: frame indices where transformations were interpolated
        """

        def _is_empty(matrix: Optional[np.ndarray]) -> bool:
            if matrix is None:
                return False
            return np.isnan(matrix).any()

        interpolated_idxs = []  # track interpolated indices
        missing_idxs = [i for i, x in enumerate(affines) if _is_empty(x)]
        if len(missing_idxs) == 0:
            return np.stack(affines), []
        if len(missing_idxs) == len(affines):
            raise VideoAligner.AlignmentError(
                "No transformations were calculated because too few keypoints were identified "
                "(probably because too few keypoints were identified)"
            )

        prev_i = None
        for i, affine in enumerate(affines):
            empty = _is_empty(affine)
            if not empty and prev_i == i - 1:
                prev_i = i
            # extrapolating empty initial frames
            elif not empty and prev_i is None:
                for j in range(i):
                    affines[j] = affines[i]
                interpolated_idxs += list(range(i))
                prev_i = i
            # interpolating empty gaps
            elif not empty and prev_i < i - 1:
                failed_idx_range = range(prev_i + 1, i)
                base_affines = np.array((affines[prev_i], affines[i]))
                affines_interpolated = VideoAligner._interpolate_affines_frame_range(base_affines, failed_idx_range)
                affines[prev_i + 1 : i] = affines_interpolated
                prev_i = i
                interpolated_idxs += failed_idx_range
            # extrapolating empty final frames
            elif empty and i == len(affines) - 1:
                for j in range(prev_i + 1, len(affines)):
                    affines[j] = affines[prev_i]
                interpolated_idxs += list(range(len(affines) + i + 1, len(affines)))
        affines = np.stack(affines)
        if affines.shape[1:] != (2, 3):
            raise VideoAligner.AlignmentError(
                f"An error occurred interpolating affine transforms for frames which had no affine "
                f"transform estimate"
            )
        return affines, interpolated_idxs

    @staticmethod
    def _interpolate_affines_frame_range(base_affines: np.ndarray, failed_idx_range: range) -> List[np.ndarray]:
        """
        Linearly interpolates affine transforms over the range of missing indices based on the
        affine matrices before and after the gap.

        Args:
            base_affines: [2, 2, 3] first matrix is pre-gap affine and second is post-gap
            failed_idx_range: range of indices for which affine matrices are to be interpolated

        Returns:
            affines_interpolated: list of interpolated affine matrices
        """
        # convert to angle for linear interpolation
        base_affines[:, 0, 0] = np.arccos(base_affines[:, 0, 0])
        base_affines[:, 0, 1] = np.arcsin(base_affines[:, 0, 1])
        base_affines[:, 1, 0] = np.arcsin(base_affines[:, 1, 0])
        base_affines[:, 1, 1] = np.arccos(base_affines[:, 1, 1])
        interp_range = [failed_idx_range[0] - 1, failed_idx_range[-1] + 1]
        interpolator = interp1d(interp_range, base_affines, axis=0)
        affines_interpolated = []
        for j in failed_idx_range:
            affine = interpolator(j)
            affine[0, 0] = np.cos(affine[0, 0])
            affine[0, 1] = np.sin(affine[0, 1])
            affine[1, 0] = np.sin(affine[1, 0])
            affine[1, 1] = np.cos(affine[1, 1])
            affines_interpolated.append(affine)
        return affines_interpolated

    @staticmethod
    def _get_euclidean_transforms(affines: np.ndarray) -> np.ndarray:
        """
        Converts affine matrix to a more interpretable and concise euclidean format
        Args:
            affines: [n_frames, 2, 3] affine matrices

        Returns:
            transforms: [n_frames, 3] representation of Euclidean transforms in the format:
                [x_translation, y_translation, rotation (radians)]
        """
        transforms = np.zeros((affines.shape[0], 3))
        transforms[:, :2] = affines[:, :, 2]
        transforms[:, 2] = np.arccos(affines[:, 0, 0])
        return transforms

    @staticmethod
    def _apply_affine(image: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Apply affine transformation to input image"""
        return cv2.warpAffine(image, affine, image.shape[::-1], flags=cv2.INTER_LINEAR)

    def _parallelize(self, func: Callable, *sequences: Sequence, **kwargs) -> Tuple:
        """Parallelizes a callable over a set of equal length sequences"""
        with Parallel(n_jobs=self.N_JOBS_PARALLEL, backend="multiprocessing") as parallel:
            r = range(len(sequences[0]))
            output = parallel(delayed(func)(*[x[i] for x in sequences], **kwargs) for i in r)
        return output

    def _parallelize_i(self, func: Callable, *sequences: Sequence, **kwargs) -> Tuple:
        """Parallelizes a callable over a set of equal length sequences, including the current index
           in the sequence as the first argument"""
        r = range(len(sequences[0]))
        return self._parallelize(func, r, *sequences, **kwargs)

    @staticmethod
    def _convert_to_array(*args: Sequence):
        """Converts an arbitrary number of arguments to numpy arrays"""
        arrays = [np.array(arg) for arg in args]
        return tuple(arrays)

    @staticmethod
    def _get_brightest_px(images: np.ndarray) -> Union[float, int]:
        brightest = np.percentile(images, VideoAligner.IMAGE_NORM_MAX_PERCENTILE)
        return brightest

    @staticmethod
    def _max_scale_images(
        images: np.ndarray, template: np.ndarray, brightest_px: float, output_dtype: type
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Max scale the video `images and `template` to the `brightest_px` value in the video"""
        max_px = VideoAligner.MAX_PIXEL_UINT8
        images = np.clip(images / brightest_px * max_px, a_min=0, a_max=max_px).astype(output_dtype)
        template = np.clip(template / brightest_px * max_px, a_min=0, a_max=max_px).astype(output_dtype)
        return images, template

    @staticmethod
    def _downsample(
        images: np.ndarray, template: np.ndarray, frame_downsample_rate: int, spatial_downsample_rate: Union[float, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Spatially and temporally downsample images for alignment process"""
        images_sample = images[::frame_downsample_rate]
        dst_size = tuple(np.array(images_sample[0].shape) // spatial_downsample_rate)
        if frame_downsample_rate != 1:
            images_sample = [cv2.pyrDown(frame, dstsize=dst_size) for frame in images_sample]
            template = cv2.pyrDown(template, dstsize=dst_size)
            images_sample = np.stack(images_sample).astype(np.uint8)
            template = template.astype(np.uint8)
        return images_sample, template


class LoResVideoAligner(VideoAligner):
    SPATIAL_DOWNSAMPLE_RATE = 2
