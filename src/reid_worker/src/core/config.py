from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "reid_worker"
    poll_interval_s: float = 0.0 # drain kafka as fast as possible, no sleep between batches
    log_every_n_messages: int = 25

    kafka_bootstrap_servers: str = "localhost:29092"
    input_topic: str = "reid_input"
    output_topic: str = "reid_output"
    consumer_group: str = "reid_worker_group"
    schema_path: str = "src/contracts/reid_input.avsc"
    output_schema_path: str = "src/contracts/reid_output.avsc"

    model_service_url: str = "http://localhost:8000"
    embedding_model: str = "osnet_ain"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embedding_dim: int = 512
    similarity_threshold: float = 0.73

    match_margin: float = 0.06

    spatial_reuse_threshold: float = 0.62

    
    color_guard_enabled: bool = False  # reverted to osnet_ain baseline; color hue guard net-negative (false-split grey clothing)
    color_conflict_veto_threshold: float = 0.83
    color_guard_min_person_evidence: int = 1  # person needs >=N color samples on the device before vetoing
    color_guard_max_frames: int = 12          # crops aggregated into one descriptor
    
    color_guard_min_frame_visibility: float = 0.5
    color_guard_max_frame_overlap: float = 0.30
    color_guard_min_reliable_frames: int = 3
    momentum: float = 0.8
    max_gallery_size: int = 8
    recent_person_reuse_enabled: bool = True
    recent_person_reuse_seconds: float = 2.5
    recent_person_reuse_min_iou: float = 0.2
    recent_person_reuse_max_center_distance_ratio: float = 0.75

    tracklet_min_entries: int = 6
    tracklet_max_entries: int = 60
    tracklet_window_seconds: float = 3.0
    tracklet_stale_seconds: float = 5.0
    
    tracklet_window_frames: int = 90
    tracklet_stale_frames: int = 150
   
    frame_clock_lifecycle_enabled: bool = False
    co_active_max_gap_frames: int = 18            # ~0.6s @ 30fps
    recent_person_reuse_max_gap_frames: int = 75  # ~2.5s @ 30fps
    recent_match_guard_max_gap_frames: int = 120  # ~4.0s @ 30fps
    tracklet_idle_flush_enabled: bool = True
    
    track_identity_memory_max_gap_frames: int = 180
    track_identity_memory_max_center_distance_ratio: float = 1.25
    
    tracklet_idle_flush_seconds: float = 0.5

    topk_k: int = 5
    topk_min_temporal_gap: int = 3
    overlap_lambda: float = 0.3
    min_high_quality_frames: int = 3
    high_quality_threshold: float = 0.55
    
    tracklet_readiness_consistency_threshold: float = 0.35

    gamma: float = 0.5
   
    v_worker_edge_floor_ratio: float = 0.85
    embedding_consensus_threshold: float = 0.66

    min_consensus_embeddings: int = 2
    
    agg_outlier_threshold: float = 0.5

    
    embedding_aggregate_max_overlap_ratio: float = 0.40
    tracklet_appearance_gate_enabled: bool = True
    tracklet_appearance_gate_min_v: float = 0.6
    tracklet_split_threshold: float = 0.55

    tracklet_appearance_gate_check_interval_frames: int = 6
   
    tracklet_identity_shift_guard_enabled: bool = True
    tracklet_identity_shift_min_entries: int = 12
    tracklet_identity_shift_min_endpoint_displacement_ratio: float = 0.55
    tracklet_identity_shift_min_size_ratio: float = 1.60
    tracklet_identity_shift_min_area_ratio: float = 2.20
    tracklet_identity_shift_anchor_min_frame_gap: int = 24
    tracklet_identity_shift_anchor_min_endpoint_displacement_ratio: float = 0.40
    tracklet_identity_shift_anchor_min_size_ratio: float = 1.45
    tracklet_identity_shift_anchor_min_area_ratio: float = 1.90

    duplicate_merge_established_min_score: float = 0.78

    duplicate_merge_canonical_bridge_min_margin: float = 0.12

    duplicate_merge_cross_device_enabled: bool = True
    duplicate_merge_cross_device_min_score: float = 0.50

    duplicate_merge_cross_device_min_margin: float = 0.12

    duplicate_merge_cross_device_max_tracklets: int = 8

    background_reconciler_interval_s: float = 0.0
    background_reconciler_max_persons: int = 50
    final_reconciler_passes: int = 3

    deterministic_processing_enabled: bool = True

    stream_quiescence_seconds: float = 20.0
    stream_finalization_timeout_seconds: float = 60.0

    max_new_identity_lag_seconds: float = 30.0

    gender_tracklet_flip_confidence: float = 0.80
    gender_tracklet_min_consecutive: int = 2
    gender_ambiguous_conflict_enabled: bool = True
    gender_ambiguous_conflict_tracklet_confidence: float = 0.70
    gender_ambiguous_conflict_max_person_confidence: float = 0.80

    occlusion_candidates_enabled: bool = True
    occlusion_candidate_min_entries: int = 2
    occlusion_candidate_min_visibility: float = 0.45
    untracked_detection_candidates_enabled: bool = True
    untracked_detection_raw_candidates_enabled: bool = False
    untracked_detection_min_confidence: float = 0.25
    untracked_detection_min_visibility: float = 0.35
    untracked_detection_max_track_iou: float = 0.20
    untracked_detection_cluster_enabled: bool = True
    untracked_detection_cluster_min_entries: int = 2
    untracked_detection_cluster_max_gap_frames: int = 60  # ~2s: rejoin a briefly-occluded person's detections (same pos+appearance) instead of fragmenting into sub-promote clusters
    untracked_detection_cluster_max_center_distance_ratio: float = 0.95
    untracked_detection_cluster_appearance_gate_enabled: bool = True
    untracked_detection_cluster_appearance_min_visibility: float = 0.55
    untracked_detection_cluster_appearance_min_similarity: float = 0.62

    untracked_detection_cluster_max_entries: int = 30
    untracked_detection_cluster_flush_after_frames: int = 36

    untracked_cluster_promote_enabled: bool = True
    untracked_cluster_promote_min_entries: int = 6
    untracked_cluster_promote_min_visibility: float = 0.65
    
    untracked_cluster_promote_min_entries_fast: int = 5
    untracked_cluster_promote_min_visibility_fast: float = 0.85
    untracked_cluster_promote_fast_min_overall_consistency: float = 0.83
   
    untracked_cluster_evidence_attach_enabled: bool = True
    untracked_cluster_evidence_attach_min_visibility: float = 0.65
    recover_stale_tracklets_enabled: bool = True
    
    fragment_recovery_enabled: bool = False
    fragment_recovery_min_fragments: int = 2
    fragment_recovery_min_total_entries: int = 5
    fragment_recovery_min_visibility: float = 0.72
    fragment_recovery_min_similarity: float = 0.62

    fragment_recovery_max_gap_frames: int = 180
    fragment_recovery_max_center_distance_ratio: float = 1.8
    
    fragment_recovery_near_gallery_threshold: float = 0.52


    promote_v_threshold: float = 0.6
    promote_consistency_threshold: float = 0.65
    synthetic_new_identity_min_overall_consistency: float = 0.75
   
    good_streak_promotion_enabled: bool = True
    good_streak_min_consecutive: int = 4
    
    new_identity_min_tracklet_len: int = 6

    tentative_max_attempts: int = 5
    
    tentative_fallback_enabled: bool = True

    update_v_threshold: float = 0.6
    update_consistency_threshold: float = 0.7
    update_min_tracklet_len: int = 5
    update_sim_threshold: float = 0.55

    gallery_update_max_overlap_ratio: float = 0.25
    gallery_update_min_overall_consistency: float = 0.80
    
    soft_match_threshold: float = 0.73

    eager_soft_match_threshold: float = 0.73


    low_visibility_threshold: float = 0.65
    low_visibility_match_threshold: float = 0.75


    scale_aux_gallery_enabled: bool = True
    scale_aux_crop_top_ratio: float = 0.48
    scale_aux_match_threshold: float = 0.66

    scale_aux_match_margin: float = 0.03

    scale_aux_full_gallery_min_score: float = 0.62

    scale_aux_min_v: float = 0.70
    scale_aux_min_consistency: float = 0.80
    scale_aux_min_tracklet_len: int = 5
    scale_aux_max_overlap_ratio: float = 0.35
   
    match_consistency_threshold: float = 0.55
   
    person_snapshot_max_overlap_ratio: float = 0.35
    current_identity_min_score: float = 0.50

    current_identity_switch_min_score: float = 0.78

    current_identity_switch_min_margin: float = 0.18

    current_identity_switch_max_current_score: float = 0.70

    max_person_identities: int = 0
    capped_identity_soft_match_threshold: float = 0.57

    near_gallery_defer_threshold: float = 0.60
   
    near_gallery_deferred_mint_max_score: float = 0.64
    
    untracked_cluster_lastresort_mint_enabled: bool = True
    untracked_cluster_lastresort_mint_min_visibility: float = 0.85
    untracked_cluster_lastresort_mint_min_entries: int = 6

    occlusion_provisional_match_enabled: bool = True
    occlusion_provisional_match_threshold: float = 0.75

    occlusion_provisional_min_margin: float = 0.03

    occlusion_provisional_short_reentry_enabled: bool = True
    occlusion_provisional_reentry_min_similarity: float = 0.55

    occlusion_provisional_reentry_max_entries: int = 8
    occlusion_provisional_reentry_max_gap_frames: int = 120
    occlusion_provisional_reentry_max_center_distance_ratio: float = 2.0

    occlusion_provisional_match_max_overlap_ratio: float = 0.40

    track_high_thresh: float = 0.7
    track_low_thresh: float = 0.35
    match_thresh: float = 0.3
    new_track_thresh: float = 0.82
    track_buffer: int = 90
    fuse_score: bool = True

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "reid_production"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    person_id_seq_key: str = "reid:seq:person_id"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"

    # Gender voting
    gender_person_threshold: float = 0.7
    
    gender_flip_threshold: float = 0.65
    attribute_flip_threshold: float = 0.85
    
    par_vote_all_extracted: bool = True
    par_min_v_score: float = 0.55
    glasses_best_frame_override_threshold: float = 0.6
    attribute_crop_top_padding_ratio: float = 0.22
    attribute_crop_side_padding_ratio: float = 0.08
    attribute_crop_bottom_padding_ratio: float = 0.04
    blocked_match_score_threshold: float = 0.9

    duplicate_track_iou_threshold: float = 0.5
    cooccurrence_guard_enabled: bool = True
    cooccurrence_guard_min_shared_frames: int = 1
    cooccurrence_guard_max_iou: float = 0.15
    cooccurrence_guard_min_center_distance_ratio: float = 0.55
    attribute_conflict_guard_enabled: bool = True
    attribute_conflict_person_confidence: float = 0.60
    attribute_conflict_tracklet_confidence: float = 0.88
    attribute_conflict_person_min_support: int = 1
    pretrack_static_filter_enabled: bool = False
    pretrack_static_filter_min_frames: int = 4
    pretrack_static_filter_max_width_px: float = 130.0
    pretrack_static_filter_max_height_px: float = 260.0
    pretrack_static_filter_max_center_drift_px: float = 6.0
    static_artifact_filter_enabled: bool = True
    static_artifact_max_mean_width_px: float = 130.0
    static_artifact_max_mean_height_px: float = 260.0
    
    static_artifact_max_path_displacement_ratio: float = 0.05
    static_artifact_max_endpoint_displacement_ratio: float = 0.02
    static_artifact_min_bbox_stability: float = 0.97
    static_artifact_min_position_stability: float = 0.97
    static_artifact_min_entries: int = 6
    
    static_artifact_boundary_contact_skip: float = 0.3
   
    static_person_filter_enabled: bool = True
    static_person_max_centroid_spread_ratio: float = 1.5
    static_person_min_tracklets: int = 2
    
    static_person_zero_motion_max_spread_x_px: float = 150.0
    static_person_zero_motion_max_spread_y_px: float = 45.0
    static_person_zero_motion_min_tracklets: int = 1
    recent_match_guard_enabled: bool = True
    recent_match_guard_seconds: float = 4.0
    recent_match_guard_min_iou: float = 0.1
    recent_match_guard_max_center_distance_ratio: float = 0.9
    gallery_update_anchor_min_score: float = 0.50

    gallery_consensus_weight: float = 0.5
    duplicate_merge_enabled: bool = True
    duplicate_merge_min_score: float = 0.90

    duplicate_merge_weak_max_tracklets: int = 2
    duplicate_merge_singleton_min_score: float = 0.90

    duplicate_merge_singleton_min_target_tracklets: int = 3
    duplicate_merge_min_margin: float = 0.10

    duplicate_merge_temporal_continuity_enabled: bool = True
    duplicate_merge_temporal_continuity_min_score: float = 0.85

    duplicate_merge_temporal_continuity_max_gap_frames: int = 15
    duplicate_merge_adjacent_fragment_enabled: bool = True
    duplicate_merge_adjacent_fragment_min_score: float = 0.70

    duplicate_merge_adjacent_fragment_max_gap_frames: int = 3
    duplicate_merge_occlusion_reentry_enabled: bool = True
    duplicate_merge_occlusion_reentry_min_score: float = 0.58

    duplicate_merge_occlusion_reentry_max_gap_frames: int = 180
    duplicate_merge_occlusion_reentry_max_center_distance_ratio: float = 2.0
    duplicate_merge_same_gender_singleton_enabled: bool = True
    duplicate_merge_same_gender_singleton_min_score: float = 0.80
    duplicate_merge_same_gender_singleton_gender_confidence: float = 0.80
    duplicate_merge_singleton_unknown_attr_min_score: float = 0.88

    duplicate_merge_attr_override_threshold: float = 0.80

    duplicate_merge_cooccurrence_override_threshold: float = 0.85

    duplicate_merge_soft_split_override_threshold: float = 0.70

    duplicate_merge_soft_split_max_weak_tracklets: int = 8
    duplicate_merge_soft_split_max_center_distance_ratio: float = 0.15
    duplicate_merge_soft_split_duplicate_iou_threshold: float = 0.75
    duplicate_merge_soft_split_duplicate_box_multitrack_min_score: float = 0.78

    duplicate_merge_soft_split_spatial_only_min_score: float = 0.50

    duplicate_merge_soft_split_spatial_only_multitrack_min_score: float = 0.60

    duplicate_merge_soft_split_spatial_only_max_center_distance_ratio: float = 0.30
    duplicate_merge_overlap_spatial_duplicate_enabled: bool = True
    duplicate_merge_overlap_spatial_duplicate_min_score: float = 0.78

    duplicate_merge_overlap_spatial_duplicate_max_gap_frames: int = 4
    duplicate_merge_overlap_spatial_duplicate_max_tracklets: int = 24
    duplicate_merge_overlap_spatial_duplicate_max_center_distance_ratio: float = 0.08
    duplicate_merge_overlap_spatial_duplicate_max_size_ratio: float = 1.25
    duplicate_merge_overlap_spatial_duplicate_max_area_ratio: float = 1.60
    duplicate_merge_trajectory_reentry_enabled: bool = True
    duplicate_merge_trajectory_reentry_min_score: float = 0.60

    duplicate_merge_trajectory_reentry_max_gap_frames: int = 240
    duplicate_merge_trajectory_reentry_max_tracklets: int = 24
    duplicate_merge_trajectory_reentry_max_center_distance_ratio: float = 0.06
    duplicate_merge_trajectory_reentry_max_size_ratio: float = 1.30
    duplicate_merge_trajectory_reentry_max_area_ratio: float = 2.00
    duplicate_merge_singleton_spatial_continuation_min_score: float = 0.30

    duplicate_merge_singleton_spatial_continuation_max_gap_frames: int = 15
    duplicate_merge_singleton_spatial_continuation_max_center_distance_ratio: float = 0.30
    duplicate_merge_singleton_spatial_continuation_max_size_ratio: float = 1.80
    duplicate_merge_singleton_spatial_continuation_max_area_ratio: float = 2.20
    duplicate_merge_adjacent_tight_continuation_enabled: bool = True
    duplicate_merge_adjacent_tight_continuation_min_score: float = 0.50

    duplicate_merge_adjacent_tight_continuation_max_gap_frames: int = 4
    duplicate_merge_adjacent_tight_continuation_max_tracklets: int = 8
    duplicate_merge_adjacent_tight_continuation_max_center_distance_ratio: float = 0.06
    duplicate_merge_adjacent_tight_continuation_max_size_ratio: float = 1.10
    duplicate_merge_adjacent_tight_continuation_max_area_ratio: float = 1.15
    duplicate_merge_boundary_weak_continuation_enabled: bool = True
    duplicate_merge_boundary_weak_continuation_min_score: float = 0.50

    duplicate_merge_boundary_weak_continuation_max_gap_frames: int = 12
    duplicate_merge_boundary_weak_continuation_max_weak_tracklets: int = 2
    duplicate_merge_boundary_weak_continuation_max_supported_tracklets: int = 8
    duplicate_merge_boundary_weak_continuation_min_center_distance_ratio: float = 0.10
    duplicate_merge_boundary_weak_continuation_max_center_distance_ratio: float = 0.32
    duplicate_merge_boundary_weak_continuation_max_size_ratio: float = 1.80
    duplicate_merge_boundary_weak_continuation_max_area_ratio: float = 2.30
    duplicate_merge_boundary_weak_continuation_max_bottom_delta_ratio: float = 0.06
    duplicate_merge_boundary_duplicate_min_score: float = 0.68

    duplicate_merge_boundary_duplicate_min_iou: float = 0.10
    duplicate_merge_boundary_duplicate_max_center_distance_ratio: float = 0.45
    duplicate_merge_ultra_continuity_min_score: float = 0.50

    duplicate_merge_ultra_continuity_max_gap_frames: int = 6
    duplicate_merge_ultra_continuity_max_center_distance_ratio: float = 0.12
    duplicate_merge_ultra_continuity_max_weak_tracklets: int = 2
    duplicate_merge_ultra_continuity_max_supported_tracklets: int = 8
    duplicate_merge_tight_spatial_reentry_enabled: bool = True
    duplicate_merge_tight_spatial_reentry_min_score: float = 0.50

    duplicate_merge_tight_spatial_reentry_max_gap_frames: int = 6
    duplicate_merge_tight_spatial_reentry_max_center_distance_ratio: float = 0.12
    duplicate_merge_tight_spatial_reentry_max_size_ratio: float = 1.15
    duplicate_merge_tight_spatial_reentry_max_area_ratio: float = 1.25
    duplicate_merge_tight_spatial_reentry_max_weak_tracklets: int = 2
    duplicate_merge_same_frame_established_duplicate_enabled: bool = True
    duplicate_merge_same_frame_established_duplicate_min_score: float = 0.50

    duplicate_merge_same_frame_established_duplicate_min_iou: float = 0.75
    duplicate_merge_same_frame_established_duplicate_max_center_distance_ratio: float = 0.05
    duplicate_merge_same_frame_established_duplicate_max_size_ratio: float = 1.15
    duplicate_merge_same_frame_established_duplicate_max_area_ratio: float = 1.25
    duplicate_merge_reentry_bridge_enabled: bool = True
    duplicate_merge_reentry_bridge_min_score: float = 0.535

    duplicate_merge_reentry_bridge_max_tracklets: int = 4
    duplicate_merge_reentry_bridge_max_supported_tracklets: int = 8
    duplicate_merge_reentry_bridge_min_gap_frames: int = 30
    duplicate_merge_reentry_bridge_max_gap_frames: int = 180
    duplicate_merge_reentry_bridge_max_center_distance_ratio: float = 0.85
    duplicate_merge_reentry_bridge_gender_confidence: float = 0.70
    duplicate_merge_reentry_bridge_min_attr_matches: int = 2
    duplicate_merge_reentry_bridge_supported_min_score: float = 0.70

    duplicate_merge_reentry_bridge_supported_min_margin: float = 0.12

    duplicate_merge_high_conf_reentry_enabled: bool = True
    duplicate_merge_high_conf_reentry_min_score: float = 0.86

    duplicate_merge_high_conf_reentry_max_tracklets: int = 8
    duplicate_merge_high_conf_reentry_min_gap_frames: int = 30
    duplicate_merge_high_conf_reentry_max_gap_frames: int = 120
    duplicate_merge_high_conf_reentry_max_center_distance_ratio: float = 0.35
    duplicate_merge_high_conf_reentry_max_size_ratio: float = 1.70
    duplicate_merge_high_conf_reentry_max_area_ratio: float = 2.80
    duplicate_merge_high_conf_reentry_attr_confidence: float = 0.65
    duplicate_merge_high_conf_reentry_min_attr_matches: int = 2
    
    duplicate_merge_scale_aware_reentry_enabled: bool = True
    duplicate_merge_scale_aware_reentry_min_score: float = 0.55

    duplicate_merge_scale_aware_reentry_max_gap_frames: int = 240
    duplicate_merge_scale_aware_reentry_max_tracklets: int = 8
    duplicate_merge_scale_aware_reentry_max_center_distance_ratio: float = 1.30
    duplicate_merge_scale_aware_reentry_max_bottom_delta_ratio: float = 0.08
    duplicate_merge_scale_aware_reentry_min_size_ratio: float = 1.00
    duplicate_merge_scale_aware_reentry_max_size_ratio: float = 2.20
    duplicate_merge_scale_aware_reentry_max_area_ratio: float = 4.00
    duplicate_merge_scale_aware_reentry_attr_confidence: float = 0.65
    duplicate_merge_scale_aware_reentry_min_attr_matches: int = 2
    duplicate_merge_supported_spatial_reentry_enabled: bool = True
    duplicate_merge_supported_spatial_reentry_min_score: float = 0.53

    duplicate_merge_supported_spatial_reentry_max_tracklets: int = 8
    duplicate_merge_supported_spatial_reentry_min_gap_frames: int = 24
    duplicate_merge_supported_spatial_reentry_max_gap_frames: int = 90
    duplicate_merge_supported_spatial_reentry_max_center_distance_ratio: float = 0.18
    duplicate_merge_supported_spatial_reentry_max_size_ratio: float = 1.20
    duplicate_merge_supported_spatial_reentry_max_area_ratio: float = 1.80
   
    duplicate_merge_clothing_reentry_enabled: bool = False
    duplicate_merge_clothing_reentry_min_score: float = 0.515

    duplicate_merge_clothing_reentry_max_weak_tracklets: int = 2
    duplicate_merge_clothing_reentry_max_supported_tracklets: int = 8
    duplicate_merge_clothing_reentry_min_gap_frames: int = 30
    duplicate_merge_clothing_reentry_max_gap_frames: int = 240
    duplicate_merge_clothing_reentry_min_attr_matches: int = 3
    duplicate_merge_clothing_reentry_attr_confidence: float = 0.70
    duplicate_merge_weak_to_supported_guard_enabled: bool = True
    duplicate_merge_weak_to_supported_min_target_tracklets: int = 5
    duplicate_merge_weak_to_supported_max_target_tracklets: int = 8
    duplicate_merge_weak_to_supported_min_score: float = 0.78

    duplicate_merge_weak_to_supported_min_margin: float = 0.08

    duplicate_merge_weak_to_supported_strong_score: float = 0.89

    duplicate_merge_weak_to_supported_strong_margin: float = 0.18

    duplicate_merge_occlusion_spatial_rejoin_enabled: bool = False
    duplicate_merge_occlusion_spatial_rejoin_min_score: float = 0.53

    duplicate_merge_occlusion_spatial_rejoin_strong_min_score: float = 0.59

    duplicate_merge_occlusion_spatial_rejoin_max_gap_frames: int = 180
    duplicate_merge_occlusion_spatial_rejoin_max_center_distance_ratio: float = 0.50
    duplicate_merge_occlusion_spatial_rejoin_tight_center_distance_ratio: float = 0.42
    duplicate_merge_occlusion_spatial_rejoin_max_size_ratio: float = 1.55
    duplicate_merge_occlusion_spatial_rejoin_max_area_ratio: float = 2.10
    duplicate_merge_occlusion_spatial_rejoin_tight_size_ratio: float = 1.10
    duplicate_merge_occlusion_spatial_rejoin_tight_area_ratio: float = 2.00
    duplicate_merge_spatial_continuation_enabled: bool = False
    duplicate_merge_spatial_continuation_min_score: float = 0.20

    duplicate_merge_spatial_continuation_max_gap_frames: int = 60
    duplicate_merge_spatial_continuation_max_center_distance_ratio: float = 0.30
   
    gender_block_sighting_confidence: float = 0.80

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="WORKER_",
        extra="ignore",
    )


settings = Settings()
