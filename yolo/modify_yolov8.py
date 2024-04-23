"""
required:
onnx
onnx-graphsurgeon

"""
import onnx_graphsurgeon as gs
import onnx
import numpy as np

if __name__ == "__main__":
  graph = gs.import_onnx(onnx.load("../models/yolo/yolov8n.onnx"))
  tensors = graph.tensors()
  nodes = graph.nodes
  outputs = graph.outputs
  boxes = gs.Variable("boxes", dtype=np.float32)
  scores = gs.Variable("scores", dtype=np.float32)
  outputs = []
  bboxes = tensors['/22/Mul_output_0']
  bboxes.dtype = np.float32
  bboxes.shape = [1, 4, 8400]
  bboxes.name = "bboxes"

  scores = tensors['/22/Sigmoid_output_0']
  scores.dtype = np.float32
  scores.shape = [1, 80, 8400]
  scores.name = "scores"

  graph.outputs = [bboxes, scores]
  graph.cleanup()
  onnx.save(gs.export_onnx(graph), "../quantize_jixiu/models/yolov8n_opt.onnx")