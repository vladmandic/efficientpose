# EfficientPose: Body Segmentation for TFJS and NodeJS

Models included in `/model-tfjs-graph-*` were converted to TFJS Graph model format from the original repository  
Models descriptors have been additionally parsed for readability

Actual model parsing implementation in `efficientpose.js` does not follow original  
and is implemented using native TFJS ops and optimized for JavaScript execution

<br><hr><br>

## Test

```shell
node efficientpose.js body.jpg
```

```js
2021-03-24 15:41:37 INFO:  efficientpose version 0.0.1
2021-03-24 15:41:37 INFO:  User: vlado Platform: linux Arch: x64 Node: v15.12.0
2021-03-24 15:41:37 INFO:  Loaded model { modelPath: 'file://models/iv/efficientpose.json', minScore: 0.2 } tensors: 955 bytes: 25643252
2021-03-24 15:41:37 INFO:  Model Signature {
  inputs: { input_res1: { name: 'input_res1', dtype: 'DT_FLOAT', tensorShape: { dim: [ { size: '1' }, { size: '600' }, { size: '600' }, { size: '3' } } } },
  outputs: { 'upscaled_confs/BiasAdd:0': { name: 'upscaled_confs/BiasAdd:0', dtype: 'DT_FLOAT', tensorShape: { dim: [ { size: '1' }, { size: '-1' }, { size: '-1' }, { size: '16' } } } }
}
2021-03-24 15:41:37 INFO:  Loaded image: body.jpg inputShape: [ 1024, 1024, 3 ] modelShape: [ 1, 600, 600, 3 ] decoded size: 3145728
2021-03-24 15:41:39 DATA:  Results: [
  { id: 0, score: 0.8234584331512451, label: 'head', xRaw: 0.4033333333333333, yRaw: 0.051666666666666666, x: 413, y: 53 },
  { id: 1, score: 0.8789138197898865, label: 'neck', xRaw: 0.4533333333333333, yRaw: 0.18166666666666667, x: 464, y: 186 },
  { id: 2, score: 0.8490188717842102, label: 'rightShoulder', xRaw: 0.395, yRaw: 0.205, x: 404, y: 210 },
  { id: 3, score: 0.8640593886375427, label: 'rightElbow', xRaw: 0.40166666666666667, yRaw: 0.3333333333333333, x: 411, y: 341 },
  { id: 4, score: 0.8743583559989929, label: 'rightWrist', xRaw: 0.4066666666666667, yRaw: 0.45666666666666667, x: 416, y: 468 },
  { id: 5, score: 0.8736196756362915, label: 'chest', xRaw: 0.46166666666666667, yRaw: 0.21166666666666667, x: 473, y: 217 },
  { id: 6, score: 0.8904648423194885, label: 'leftShoulder', xRaw: 0.5283333333333333, yRaw: 0.215, x: 541, y: 220 },
  { id: 7, score: 0.9026476144790649, label: 'leftElbow', xRaw: 0.525, yRaw: 0.3616666666666667, x: 538, y: 370 },
  { id: 8, score: 0.7956844568252563, label: 'leftWrist', xRaw: 0.47333333333333333, yRaw: 0.49166666666666664, x: 485, y: 503 },
  { id: 9, score: 0.8972961902618408, label: 'pelvis', xRaw: 0.5066666666666667, yRaw: 0.45666666666666667, x: 519, y: 468 },
  { id: 10, score: 0.807637631893158, label: 'rightHip', xRaw: 0.4666666666666667, yRaw: 0.45666666666666667, x: 478, y: 468 },
  { id: 11, score: 0.8232259750366211, label: 'rightKnee', xRaw: 0.47833333333333333, yRaw: 0.63, x: 490, y: 645 },
  { id: 12, score: 0.9226986765861511, label: 'rightAnkle', xRaw: 0.43833333333333335, yRaw: 0.79, x: 449, y: 809 },
  { id: 13, score: 0.7791210412979126, label: 'leftHip', xRaw: 0.545, yRaw: 0.4533333333333333, x: 558, y: 464 },
  { id: 14, score: 0.8537712097167969, label: 'leftKnee', xRaw: 0.5883333333333334, yRaw: 0.65, x: 602, y: 666 },
  { id: 15, score: 0.8724350333213806, label: 'leftAnkle', xRaw: 0.6016666666666667, yRaw: 0.8433333333333334, x: 616, y: 864 },
]
2021-03-24 15:41:39 STATE:  Created output image: outputs/body.jpg size: [ 1024, 1024 ]
```

<br><hr><br>

## Conversion Notes

Original: <https://github.com/daniegr/EfficientPose>

### Requirements

Edit `requirements.txt` to remove specific version pinning and install required packages:

```shell
sudo apt install libmediainfo-dev
pip install -r requirements.txt
pip install tensorflowjs
```

### Test

Edit `track.py` to fix tensor names:

```python
    # TensorFlow
    elif framework in ['tensorflow', 'tf']:
        output_tensor = model.graph.get_tensor_by_name('upscaled_confs/BiasAdd:0')
        if lite:
            batch_outputs = model.run(output_tensor, {'input_1_0:0': batch})            
        else:
            batch_outputs = model.run(output_tensor, {'input_res1:0': batch})
```

Run test:

```shell
python track.py --path=body.jpg --model=II_Lite --framework=tensorflow --visualize --store
```

### Convert

From TensorFlow Frozen model to TFJS Graph model:

```shell
tensorflowjs_converter \
--input_format tf_frozen_model \
--output_format tfjs_graph_model \
--strip_debug_ops=* \
--weight_shard_size_bytes=16777216 \
--output_node_names='upscaled_confs/BiasAdd:0' \
tensorflow/EfficientPoseII_LITE.pb \
tfjs/ii-lite
```

After conversion, lets add correct model signature in `model.json`

```json
  "signature": {
      "inputs": { "input_1_0": { "name": "input_1_0", "dtype": "DT_FLOAT", "tensorShape":{"dim":[{"size":"1"},{"size":"368"},{"size":"368"},{"size":"3"}]} } },
      "outputs": { "upscaled_confs/BiasAdd:0": { "name": "upscaled_confs/BiasAdd:0", "dtype": "DT_FLOAT", "tensorShape":{"dim":[{"size":"1"},{"size":"-1"},{"size":"-1"},{"size":"16"}]} } }
  },
```
