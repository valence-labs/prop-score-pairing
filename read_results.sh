#!/bin/bash
# first argument is the line number

echo "Classifier EOT"

for i in {1..10}
do
   in=$(sed -n "$1"p results/BallsClassifier_CLASSIFIERBALLS$i.ckptBALLS_EOT.csv)
   arrIN=(${in//,/ })
   echo ${arrIN[$2]},
done

echo "Classifier sNN"

for i in {1..10}
do
   in=$(sed -n "$1"p results/BallsClassifier_CLASSIFIERBALLS$i.ckptBALLS_kNN.csv)
   arrIN=(${in//,/ })
   echo ${arrIN[$2]},
done

echo "VAE EOT"

for i in {1..10}
do
   in=$(sed -n "$1"p results/ImageVAEModule_VAEBALLS$i.ckptBALLS_EOT.csv)
   arrIN=(${in//,/ })
   echo ${arrIN[$2]},
done

echo "VAE SNN"

for i in {1..10}
do
   in=$(sed -n "$1"p results/ImageVAEModule_VAEBALLS$i.ckptBALLS_kNN.csv)
   arrIN=(${in//,/ })
   echo ${arrIN[$2]},
done