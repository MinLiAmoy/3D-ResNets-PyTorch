python main.py --root_path ~/Data --video_path UCF101_JPG_FGSM --annotation_path ucf101_01.json --result_path results --dataset ucf101 --model resnet --resume_path results/resnet-18-kinetics-ucf101_split1.pth --model_depth 18 --n_classes 101 --batch_size 32 --n_threads 4 --checkpoint 5 --resnet_shortcut A --no_train --test

attack:
python main.py --root_path ~/Data --video_path UCF101_JPG_FGSM --annotation_path ucf101_01.json --result_path results --dataset ucf101 --model resnet --resume_path results/resnet-18-kinetics-ucf101_split1.pth --model_depth 18 --n_classes 101 --batch_size 32 --n_threads 4 --checkpoint 5 --resnet_shortcut A --no_train --adv_attack

