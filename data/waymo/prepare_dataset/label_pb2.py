# Copyright 2019 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: waymo_open_dataset/label.proto

from google.protobuf import descriptor_pb2
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import reflection as _reflection
from google.protobuf import message as _message
from google.protobuf import descriptor as _descriptor
import sys
_b = sys.version_info[0] < 3 and (
    lambda x: x) or (
        lambda x: x.encode('latin1'))
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='waymo_open_dataset/label.proto',
    package='waymo.open_dataset',
    syntax='proto2',
    serialized_pb=_b('\n\x1ewaymo_open_dataset/label.proto\x12\x12waymo.open_dataset\"\x8f\x06\n\x05Label\x12*\n\x03\x62ox\x18\x01 \x01(\x0b\x32\x1d.waymo.open_dataset.Label.Box\x12\x34\n\x08metadata\x18\x02 \x01(\x0b\x32\".waymo.open_dataset.Label.Metadata\x12,\n\x04type\x18\x03 \x01(\x0e\x32\x1e.waymo.open_dataset.Label.Type\x12\n\n\x02id\x18\x04 \x01(\t\x12M\n\x1a\x64\x65tection_difficulty_level\x18\x05 \x01(\x0e\x32).waymo.open_dataset.Label.DifficultyLevel\x12L\n\x19tracking_difficulty_level\x18\x06 \x01(\x0e\x32).waymo.open_dataset.Label.DifficultyLevel\x12\x1f\n\x17num_lidar_points_in_box\x18\x07 \x01(\x05\x1a\xbf\x01\n\x03\x42ox\x12\x10\n\x08\x63\x65nter_x\x18\x01 \x01(\x01\x12\x10\n\x08\x63\x65nter_y\x18\x02 \x01(\x01\x12\x10\n\x08\x63\x65nter_z\x18\x03 \x01(\x01\x12\x0e\n\x06length\x18\x05 \x01(\x01\x12\r\n\x05width\x18\x04 \x01(\x01\x12\x0e\n\x06height\x18\x06 \x01(\x01\x12\x0f\n\x07heading\x18\x07 \x01(\x01\"B\n\x04Type\x12\x10\n\x0cTYPE_UNKNOWN\x10\x00\x12\x0b\n\x07TYPE_3D\x10\x01\x12\x0b\n\x07TYPE_2D\x10\x02\x12\x0e\n\nTYPE_AA_2D\x10\x03\x1aN\n\x08Metadata\x12\x0f\n\x07speed_x\x18\x01 \x01(\x01\x12\x0f\n\x07speed_y\x18\x02 \x01(\x01\x12\x0f\n\x07\x61\x63\x63\x65l_x\x18\x03 \x01(\x01\x12\x0f\n\x07\x61\x63\x63\x65l_y\x18\x04 \x01(\x01\"`\n\x04Type\x12\x10\n\x0cTYPE_UNKNOWN\x10\x00\x12\x10\n\x0cTYPE_VEHICLE\x10\x01\x12\x13\n\x0fTYPE_PEDESTRIAN\x10\x02\x12\r\n\tTYPE_SIGN\x10\x03\x12\x10\n\x0cTYPE_CYCLIST\x10\x04\"8\n\x0f\x44ifficultyLevel\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0b\n\x07LEVEL_1\x10\x01\x12\x0b\n\x07LEVEL_2\x10\x02\"2\n\x0ePolygon2dProto\x12\t\n\x01x\x18\x01 \x03(\x01\x12\t\n\x01y\x18\x02 \x03(\x01\x12\n\n\x02id\x18\x03 \x01(\t')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)


_LABEL_BOX_TYPE = _descriptor.EnumDescriptor(
    name='Type',
    full_name='waymo.open_dataset.Label.Box.Type',
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name='TYPE_UNKNOWN', index=0, number=0,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='TYPE_3D', index=1, number=1,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='TYPE_2D', index=2, number=2,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='TYPE_AA_2D', index=3, number=3,
            options=None,
            type=None),
    ],
    containing_type=None,
    options=None,
    serialized_start=536,
    serialized_end=602,
)
_sym_db.RegisterEnumDescriptor(_LABEL_BOX_TYPE)

_LABEL_TYPE = _descriptor.EnumDescriptor(
    name='Type',
    full_name='waymo.open_dataset.Label.Type',
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name='TYPE_UNKNOWN', index=0, number=0,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='TYPE_VEHICLE', index=1, number=1,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='TYPE_PEDESTRIAN', index=2, number=2,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='TYPE_SIGN', index=3, number=3,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='TYPE_CYCLIST', index=4, number=4,
            options=None,
            type=None),
    ],
    containing_type=None,
    options=None,
    serialized_start=684,
    serialized_end=780,
)
_sym_db.RegisterEnumDescriptor(_LABEL_TYPE)

_LABEL_DIFFICULTYLEVEL = _descriptor.EnumDescriptor(
    name='DifficultyLevel',
    full_name='waymo.open_dataset.Label.DifficultyLevel',
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name='UNKNOWN', index=0, number=0,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='LEVEL_1', index=1, number=1,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='LEVEL_2', index=2, number=2,
            options=None,
            type=None),
    ],
    containing_type=None,
    options=None,
    serialized_start=782,
    serialized_end=838,
)
_sym_db.RegisterEnumDescriptor(_LABEL_DIFFICULTYLEVEL)


_LABEL_BOX = _descriptor.Descriptor(
    name='Box',
    full_name='waymo.open_dataset.Label.Box',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='center_x', full_name='waymo.open_dataset.Label.Box.center_x', index=0,
            number=1, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='center_y', full_name='waymo.open_dataset.Label.Box.center_y', index=1,
            number=2, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='center_z', full_name='waymo.open_dataset.Label.Box.center_z', index=2,
            number=3, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='length', full_name='waymo.open_dataset.Label.Box.length', index=3,
            number=5, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='width', full_name='waymo.open_dataset.Label.Box.width', index=4,
            number=4, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='height', full_name='waymo.open_dataset.Label.Box.height', index=5,
            number=6, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='heading', full_name='waymo.open_dataset.Label.Box.heading', index=6,
            number=7, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
        _LABEL_BOX_TYPE,
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=411,
    serialized_end=602,
)

_LABEL_METADATA = _descriptor.Descriptor(
    name='Metadata',
    full_name='waymo.open_dataset.Label.Metadata',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='speed_x', full_name='waymo.open_dataset.Label.Metadata.speed_x', index=0,
            number=1, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='speed_y', full_name='waymo.open_dataset.Label.Metadata.speed_y', index=1,
            number=2, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='accel_x', full_name='waymo.open_dataset.Label.Metadata.accel_x', index=2,
            number=3, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='accel_y', full_name='waymo.open_dataset.Label.Metadata.accel_y', index=3,
            number=4, type=1, cpp_type=5, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=604,
    serialized_end=682,
)

_LABEL = _descriptor.Descriptor(
    name='Label',
    full_name='waymo.open_dataset.Label',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='box', full_name='waymo.open_dataset.Label.box', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='metadata', full_name='waymo.open_dataset.Label.metadata', index=1,
            number=2, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='type', full_name='waymo.open_dataset.Label.type', index=2,
            number=3, type=14, cpp_type=8, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='id', full_name='waymo.open_dataset.Label.id', index=3,
            number=4, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='detection_difficulty_level', full_name='waymo.open_dataset.Label.detection_difficulty_level', index=4,
            number=5, type=14, cpp_type=8, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='tracking_difficulty_level', full_name='waymo.open_dataset.Label.tracking_difficulty_level', index=5,
            number=6, type=14, cpp_type=8, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='num_lidar_points_in_box', full_name='waymo.open_dataset.Label.num_lidar_points_in_box', index=6,
            number=7, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[_LABEL_BOX, _LABEL_METADATA, ],
    enum_types=[
        _LABEL_TYPE,
        _LABEL_DIFFICULTYLEVEL,
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=55,
    serialized_end=838,
)


_POLYGON2DPROTO = _descriptor.Descriptor(
    name='Polygon2dProto',
    full_name='waymo.open_dataset.Polygon2dProto',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='x', full_name='waymo.open_dataset.Polygon2dProto.x', index=0,
            number=1, type=1, cpp_type=5, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='y', full_name='waymo.open_dataset.Polygon2dProto.y', index=1,
            number=2, type=1, cpp_type=5, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='id', full_name='waymo.open_dataset.Polygon2dProto.id', index=2,
            number=3, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=840,
    serialized_end=890,
)

_LABEL_BOX.containing_type = _LABEL
_LABEL_BOX_TYPE.containing_type = _LABEL_BOX
_LABEL_METADATA.containing_type = _LABEL
_LABEL.fields_by_name['box'].message_type = _LABEL_BOX
_LABEL.fields_by_name['metadata'].message_type = _LABEL_METADATA
_LABEL.fields_by_name['type'].enum_type = _LABEL_TYPE
_LABEL.fields_by_name['detection_difficulty_level'].enum_type = _LABEL_DIFFICULTYLEVEL
_LABEL.fields_by_name['tracking_difficulty_level'].enum_type = _LABEL_DIFFICULTYLEVEL
_LABEL_TYPE.containing_type = _LABEL
_LABEL_DIFFICULTYLEVEL.containing_type = _LABEL
DESCRIPTOR.message_types_by_name['Label'] = _LABEL
DESCRIPTOR.message_types_by_name['Polygon2dProto'] = _POLYGON2DPROTO

Label = _reflection.GeneratedProtocolMessageType('Label', (_message.Message,), dict(

    Box=_reflection.GeneratedProtocolMessageType('Box', (_message.Message,), dict(
        DESCRIPTOR=_LABEL_BOX,
        __module__='waymo_open_dataset.label_pb2'
        # @@protoc_insertion_point(class_scope:waymo.open_dataset.Label.Box)
    )),

    Metadata=_reflection.GeneratedProtocolMessageType('Metadata', (_message.Message,), dict(
        DESCRIPTOR=_LABEL_METADATA,
        __module__='waymo_open_dataset.label_pb2'
        # @@protoc_insertion_point(class_scope:waymo.open_dataset.Label.Metadata)
    )),
    DESCRIPTOR=_LABEL,
    __module__='waymo_open_dataset.label_pb2'
    # @@protoc_insertion_point(class_scope:waymo.open_dataset.Label)
))
_sym_db.RegisterMessage(Label)
_sym_db.RegisterMessage(Label.Box)
_sym_db.RegisterMessage(Label.Metadata)

Polygon2dProto = _reflection.GeneratedProtocolMessageType('Polygon2dProto', (_message.Message,), dict(
    DESCRIPTOR=_POLYGON2DPROTO,
    __module__='waymo_open_dataset.label_pb2'
    # @@protoc_insertion_point(class_scope:waymo.open_dataset.Polygon2dProto)
))
_sym_db.RegisterMessage(Polygon2dProto)


# @@protoc_insertion_point(module_scope)