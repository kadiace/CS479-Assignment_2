python torch_nerf/runners/render.py +log_dir=outputs/seed_100_f +render_test_views=False
python scripts/utils/create_video.py —img_dir outputs/seed_100_f/render/video/* —vid_title lego-seed_final
python torch_nerf/runners/render.py +log_dir=outputs/seed_100_f +render_test_views=True
python torch_nerf/runners/evaluate.py outputs/seed_100_f/render/test_views/* ./data/nerf_synthetic/lego/test > outputs/seed_100_f/render/test_views/metric_evaluation.txt