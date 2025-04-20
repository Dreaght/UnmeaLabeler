# UnmeaLabeler

A tool to convert the Food-101 dataset into a YOLO-compatible structure for object detection, and to manually label bounding boxes using a PyQT GUI.

## Arguments
`-d` `--dataset`: Food-101 dataset path

`-m` `--label-studio-path`: Migrate from label studio (Specify a path)

`-t` `--train-sample-size`: Training sample size

`-v` `--val-sample-size`: Validation sample size

`-s` `--seed`: Seed

`-p` `--purge`: Purge dataset before starting (0 / 1)

`-r` `--review`: Review all the dataset starting (even labeled) (0 / 1)
