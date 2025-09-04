
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
    python process_data.py GoogleNews-vectors-negative300.bin $d.p ./data/$d/validation.csv ./data/$d/train.csv

    echo "CNN-rand"
    python conv_net_sentence.py $d.p -nonstatic -rand
    echo "CNN-static"
    python conv_net_sentence.py $d.p -nonstatic -word2vec
    echo "CNN-non-static"
    python conv_net_sentence.py $d.p -static -word2vec
    echo "CNN-multichannel"
    # not impmlemtend
    # python conv_net_sentence.py $d.p -static -rand
done










