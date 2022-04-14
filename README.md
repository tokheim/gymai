python run -g Breakout-v0

python run.py -c configs/spaceinvader.yaml --render --test -l models/spaceinvaders-lowent.h5
python run.py -c configs/breakout-worker.yaml --render --test -l models/breakout-lowent-cont.h5
python run.py -c configs/breakout-worker.yaml --render --test -l models/breakout-savefix.h5
python run.py -c configs/cartpole.yaml --render --test -l models/cartpole.h5
python run.py -c configs/lunarlander.yaml --render --test -l models/lunarlander-cont.h5
