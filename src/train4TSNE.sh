docker run --user root -it --rm --name pig_org_asf_predict_attribution_train_lorcy1 \
            -v /data2/hy_data_mining/lorcy_6301/:/src \
            -v /data2/hy_data_mining/lorcy_6301/data:/data \
            -v /data2/hy_data_mining/lorcy_6301/log:/log \
            -v /etc/localtime:/etc/localtime:ro \
            --cpus 24 \
            --memory 60g \
            --gpus "device=4" \
            --shm-size=60g \
            graph_attribution_inference:cuda-11.7 \
            /bin/bash -c "cd /src/src && python train4TSNE.py"
