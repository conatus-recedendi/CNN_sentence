
# train

# mr, sst-1, sst-2, subj, trec, cr, mpqa

DATA=(
    mr
    sst-1
    sst-2
    subj
    trec
    cr
    mpqa
)

cd ..
for d in ${DATA[@]}; do
    echo "Processing $d dataset..."
    # python process_data.py GoogleNews-vectors-negative300.bin $d.p ./data/$d/validation.csv ./data/$d/train.csv
    python process_data.py GoogleNews-vectors-negative300.bin $d.p ./data/$d/train.csv ./data/$d/validation.csv 

    echo "CNN-rand"
    python conv_net_sentence_pytorch.py $d.p nonstatic rand --test-file ./data/$d/test.csv --epochs 25 --lr 0.5
    echo "CNN-static"
    python conv_net_sentence_pytorch.py $d.p static word2vec --test-file ./data/$d/test.csv --epochs 25 --lr 0.5
    echo "CNN-non-static"
    python conv_net_sentence_pytorch.py $d.p nonstatic word2vec --test-file ./data/$d/test.csv --epochs 25 --lr 0.5
    echo "CNN-multichannel"
    python conv_net_sentence_pytorch.py $d.p multichannel word2vec --test-file ./data/$d/test.csv --epochs 25 --lr 0.5

done










