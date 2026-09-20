export IMAGE=ms-swift-npu-arm-elastic:4.5.2
docker run -it -p 5092:5092 \
    --shm-size=128g \
    --name swift_test_4.5.2 \
    --device=/dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
	  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
	  -v /var/log/npu/:/var/log/npu/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /mnt/data2/lyk/traininfer/models:/mnt/data2/lyk/traininfer/models \
    -v /mnt/data2/datasets:/mnt/data2/datasets \
    -v /mnt/data2/lyk/traininfer/ms-swift/ms-swift-4.5.2/ms-swift-4.5.2:/mnt/data2/lyk/traininfer/ms-swift/ms-swift-4.5.2/ms-swift-4.5.2 \
    $IMAGE bash

