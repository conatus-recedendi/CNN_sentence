
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
done








