# Update the vllm-ascend image
export IMAGE=modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.23.0-modelscope1.38.1-swift4.4.1
docker run -itd -p 9065:7860 -p 9066:7861 \
    --shm-size=128g \
    --name swift_4.5.2 \
    --gpus '"device=1,2,3"' \
	  -v /data/AI/lyk/ms-swift/ms-swift-4.5.2:/data/AI/lyk/ms-swift/ms-swift-4.5.2 \
	  -v /data/AI/models:/data/AI/models \
    $IMAGE bash