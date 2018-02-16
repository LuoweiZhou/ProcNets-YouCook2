#!/bin/bash
# A script to extract frames from video for the entire dataset
# Please refer to the dataset readme file for directory structure

argu1="$1"

chmod +x ./video2frame_yc2.sh

source_folder="/z/tmp/luozhou/YouCookII_data/train"
target_folder="/z/tmp/luozhou/YouCookII_data/train_frames"

# list the processed videos
cd $target_folder
count=0
for i in $(ls); do
    cd $i
    for j in $(ls); do
        processed[$count]=$j
        let count=$count+1
    done
    cd ../
done

cd $source_folder

# sample a specified number of frames for all the videos in the dataset
# do data augmentation (x10) for the training data
for i in $(ls); do
    if [[ "${i:0:2}" == $argu1 ]]; then
      mkdir $target_folder/$i
      cd $i
      for j in $(ls); do    
         let videolen=11
         if [[ "${processed[@]}" =~ ${j:0:videolen} ]]; then
             printf '[INFO] %s has already been processed!\n' "${j:0:videolen}"
         else
             mkdir $target_folder/$i/${j:0:videolen}
             python /z/home/luozhou/subsystem/YouCookII/script/videosample.py -i ./$j -o $target_folder/$i/${j:0:videolen}/ -n 500 -r 10 # for training data, w/ data augmentation
             # python /z/home/luozhou/subsystem/YouCookII/script/videosample.py -i ./$j -o $target_folder/$i/${j:0:videolen}/ -n 500 -r 1 # for validation/testing data, w/o data augmentation
         fi
      done
      cd $source_folder
    fi
done
