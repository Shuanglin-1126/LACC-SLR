import argparse


def get_args():
    parser = argparse.ArgumentParser('slr', add_help=False)
    # deivce
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")

    # model
    parser.add_argument("--emd_dim", default=768, type=int)
    parser.add_argument("--drop_out", default=0.1, type=float)
    parser.add_argument("--backbone_net", default='mvit_s_two', type=str)
    parser.add_argument("--shape", type=int, nargs="+", default=[224, 224], help="input image size (height, width)")

    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=16,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Model inference dtype"
    )

    # dataset
    parser.add_argument("--file_test", default=r'path to fataset file')
    parser.add_argument("--input", type=str, default=r"path to image dataset dir", help="Image/Video file")
    parser.add_argument("--keypoints_body_path", type=str, default=r"path to skeleton dataset dir")
    parser.add_argument("--dataset", default='wlasl100', type=str)
    parser.add_argument("--datadir", default=r'path to image dataset dir', type=str)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--frames_per_group", default=1, type=int)
    parser.add_argument("--num_clips", default=1, type=int)
    parser.add_argument("--num_class", default=2100, type=int)
    parser.add_argument("--modality", default='rgb', type=str)
    parser.add_argument("--dense_sampling", default=False, type=bool)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--resume_path", type=str, default=r'path to trained checkpoint')

    return parser.parse_args()