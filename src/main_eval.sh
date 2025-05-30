docker run --user root -it --rm --name pig_org_asf_predict_attribution_train_lorcy \
            -v /data2/hy_data_mining/lorcy_630/:/src \
            -v /data2/hy_data_mining/lorcy_630/data:/data \
            -v /data2/hy_data_mining/lorcy_630/log:/log \
            -v /etc/localtime:/etc/localtime:ro \
            --cpus 24 \
            --memory 60g \
            --gpus "device=4" \
            --shm-size=60g \
            graph_attribution_inference:cuda-11.7 \
            /bin/bash -c "cd /src && python src/abortion_abnormal_evaluator.py \
	    --predict_running_dt_end '2024-06-13' \
	    --train_running_dt_end '2024-05-15' "
