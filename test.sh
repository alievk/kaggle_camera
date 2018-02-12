B=32
opt="--checkpoint best --tta"

python train.py $opt --mode predict_test -i 512 -b $B --model densenet121 -r runs/densenet_ismanip &&
python train.py $opt --mode predict_test -i 512 -b $B --model resnet101 -r runs/resnet101_ismanip &&
python train.py $opt --mode predict_test -i 512 -b $B --model resnet50 -r runs/resnet50_ismanip &&
python train.py $opt --mode predict_test -i 512 -b $B --model densenet121 -r runs/densenet121_final &&
python train.py $opt --mode predict_test -i 512 -b $B --model resnet101 -r runs/resnet101_final &&
python train.py $opt --mode predict_test -i 512 -b $B --model resnet50 -r runs/resnet50_final
