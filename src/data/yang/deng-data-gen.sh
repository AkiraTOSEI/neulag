#!/bin/bash
python data_gen.py
mv dataIn/train_*.csv ../../../Data/Deng/
mv dataIn/test_*.csv ../../../Data/Deng/
rm -rf dataIn
