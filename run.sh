python exp_runner.py --case $1
python export_mask.py --case $1
python export_texture.py --case $1
python init_shape.py --case $1
python optim_transparent.py --case $1