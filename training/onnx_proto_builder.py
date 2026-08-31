"""
ONNX ModelProto Pure-Python Builder for SharedActor (MLP: Linear(115, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, 19)).
"""

import struct
import io
import zipfile
import os

def encode_varint(val: int) -> bytes:
    buf = bytearray()
    while val > 0x7F:
        buf.append((val & 0x7F) | 0x80)
        val >>= 7
    buf.append(val & 0x7F)
    return bytes(buf)

def field_varint(field_num: int, val: int) -> bytes:
    tag = (field_num << 3) | 0
    return encode_varint(tag) + encode_varint(val)

def field_bytes(field_num: int, data: bytes) -> bytes:
    tag = (field_num << 3) | 2
    return encode_varint(tag) + encode_varint(len(data)) + data

def field_string(field_num: int, text: str) -> bytes:
    return field_bytes(field_num, text.encode("utf-8"))

def field_msg(field_num: int, msg_bytes: bytes) -> bytes:
    return field_bytes(field_num, msg_bytes)

def field_float(field_num: int, val: float) -> bytes:
    tag = (field_num << 3) | 5
    return encode_varint(tag) + struct.pack("<f", val)

# ONNX protobuf message builders

def build_tensor_shape_proto(dims):
    # repeated Dimension dim = 1;
    res = bytearray()
    for d in dims:
        dim_msg = bytearray()
        if isinstance(d, int):
            # int64 dim_value = 1;
            dim_msg += field_varint(1, d)
        elif isinstance(d, str):
            # string dim_param = 2;
            dim_msg += field_string(2, d)
        res += field_msg(1, bytes(dim_msg))
    return bytes(res)

def build_type_proto_tensor(elem_type, shape_dims):
    # Tensor tensor_type = 1;
    # inside Tensor: int32 elem_type = 1; TensorShapeProto shape = 2;
    tensor_msg = bytearray()
    tensor_msg += field_varint(1, elem_type)
    if shape_dims is not None:
        shape_msg = build_tensor_shape_proto(shape_dims)
        tensor_msg += field_msg(2, shape_msg)
    
    type_proto = bytearray()
    type_proto += field_msg(1, bytes(tensor_msg))
    return bytes(type_proto)

def build_value_info_proto(name, elem_type, shape_dims):
    # string name = 1; TypeProto type = 2;
    res = bytearray()
    res += field_string(1, name)
    res += field_msg(2, build_type_proto_tensor(elem_type, shape_dims))
    return bytes(res)

def build_tensor_proto(name, dims, elem_type, raw_data_bytes):
    # repeated int64 dims = 1;
    # int32 data_type = 2;
    # string name = 8;
    # bytes raw_data = 9;
    res = bytearray()
    for d in dims:
        res += field_varint(1, d)
    res += field_varint(2, elem_type)
    res += field_string(8, name)
    res += field_bytes(9, raw_data_bytes)
    return bytes(res)

def build_attribute_proto_float(name, val):
    # string name = 1; float f = 2; AttributeType type = 20 (FLOAT = 1);
    res = bytearray()
    res += field_string(1, name)
    res += field_float(2, val)
    res += field_varint(20, 1)
    return bytes(res)

def build_attribute_proto_int(name, val):
    # string name = 1; int64 i = 3; AttributeType type = 20 (INT = 2);
    res = bytearray()
    res += field_string(1, name)
    res += field_varint(3, val)
    res += field_varint(20, 2)
    return bytes(res)

def build_node_proto(op_type, inputs, outputs, name="", attributes=None):
    # repeated string input = 1;
    # repeated string output = 2;
    # string name = 3;
    # string op_type = 4;
    # repeated AttributeProto attribute = 5;
    res = bytearray()
    for inp in inputs:
        res += field_string(1, inp)
    for out in outputs:
        res += field_string(2, out)
    if name:
        res += field_string(3, name)
    res += field_string(4, op_type)
    if attributes:
        for attr in attributes:
            res += field_msg(5, attr)
    return bytes(res)

def build_graph_proto(name, nodes, inputs, outputs, initializers):
    # repeated NodeProto node = 1;
    # string name = 2;
    # repeated TensorProto initializer = 5;
    # repeated ValueInfoProto input = 11;
    # repeated ValueInfoProto output = 12;
    res = bytearray()
    for node in nodes:
        res += field_msg(1, node)
    res += field_string(2, name)
    for init in initializers:
        res += field_msg(5, init)
    for inp in inputs:
        res += field_msg(11, inp)
    for out in outputs:
        res += field_msg(12, out)
    return bytes(res)

def build_operator_set_id_proto(domain, version):
    # string domain = 1; int64 version = 2;
    res = bytearray()
    if domain:
        res += field_string(1, domain)
    res += field_varint(2, version)
    return bytes(res)

def build_model_proto(graph_proto, opset_version=17, producer_name="GMN-Football-3"):
    # int64 ir_version = 1 (IR_VERSION = 8);
    # repeated OperatorSetIdProto opset_import = 8;
    # string producer_name = 2;
    # GraphProto graph = 7;
    res = bytearray()
    res += field_varint(1, 8) # ir_version 8
    res += field_string(2, producer_name)
    res += field_msg(7, graph_proto)
    res += field_msg(8, build_operator_set_id_proto("", opset_version))
    return bytes(res)
