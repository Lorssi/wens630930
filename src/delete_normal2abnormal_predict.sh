docker run --user root -it --rm --name pig_org_asf_predict_attribution_train_lorcy \
            -v /data2/hy_data_mining/lorcy_630/:/src \
            -v /data2/hy_data_mining/lorcy_630/data:/data \
            -v /data2/hy_data_mining/lorcy_630/log:/log \
            -v /etc/localtime:/etc/localtime:ro \
            --cpus 24 \
            --memory 60g \
            --gpus "device=5" \
            --shm-size=60g \
            graph_attribution_inference:cuda-11.7 \
            /bin/bash -c "cd /src/src && python delete_normal2abnormal_predict.py"
