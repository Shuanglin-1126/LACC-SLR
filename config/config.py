import argparse


def get_args():
    parser = argparse.ArgumentParser('slr', add_help=False)
    # deivce
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")

    # optimizer
    parser.add_argument("--optim", default='adamw', type=str)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--minlr", default=1e-6, type=float)
    parser.add_argument('--beta1', default=0.9, type=float, metavar='M', help='beta1')
    parser.add_argument('--beta2', default=0.999, type=float, metavar='M', help='beta2')
    parser.add_argument('--weight_decay', default=5e-2, type=float)   # 1e-4

    # model
    parser.add_argument("--emd_dim", default=768, type=int)
    parser.add_argument("--drop_out", default=0.1, type=float)
    parser.add_argument("--backbone_net", default='mvit_s_two_layer', type=str)
    parser.add_argument("--fused_posi", default=1, type=int)
    parser.add_argument("--shape", type=int, nargs="+", default=[224, 224], help="input image size (height, width)")

    # training
    parser.add_argument("--epoch", default=40, type=int)
    parser.add_argument('--accumulation_steps', default=1, type=int)
    parser.add_argument("--warm_epoch", default=3, type=int)
    parser.add_argument("--save_model_fre", default=20, type=int)
    parser.add_argument(
        "--output_root",
        type=str,
        default=r"E:\chexiao\projects\output\output_mvits_fused_1",
        help="root of the output img file. "
             "Default not saving the visualization images.",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=12,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Model inference dtype"
    )

    # dataset
    parser.add_argument("--file_train", default=r'D:\zhaoyizhe\WLASL\origin_frames\wlasl100train_all_frame.txt')
    parser.add_argument("--file_val", default=r'D:\zhaoyizhe\WLASL\origin_frames\wlasl100val_all_frame.txt')
    parser.add_argument("--file_test", default=r'D:\zhaoyizhe\WLASL\origin_frames\wlasl100test_all_frame.txt')
    parser.add_argument("--input", type=str, default=r"D:\zhaoyizhe\WLASL\origin_frames", help="Image/Video file")
    parser.add_argument("--keypoints_body_path", type=str, default=r"E:\SLR_dataset\wlasl\body_keypoint")
    parser.add_argument("--dataset", default='wlasl', type=str)
    parser.add_argument("--datadir", default=r'D:\zhaoyizhe\WLASL\origin_frames', type=str)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--frames_per_group", default=1, type=int)
    parser.add_argument("--num_clips", default=1, type=int)
    parser.add_argument("--num_class", default=100, type=int)
    parser.add_argument("--modality", default='rgb', type=str)
    parser.add_argument("--dense_sampling", default=False, type=bool)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resume_path", type=str, default=r'E:\chexiao\projects\output1\output_mvits_pose_100\best_model.pth')

    return parser.parse_args()