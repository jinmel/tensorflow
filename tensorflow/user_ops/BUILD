# Description:
# An example for custom op and kernel defined as a TensorFlow plugin.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "ackermann_op.so",
    srcs = ["ackermann_op.cc"],
)

tf_py_test(
    name = "ackermann_test",
    size = "small",
    srcs = ["ackermann_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    data = [":ackermann_op.so"],
)

tf_custom_op_library(
    name = "duplicate_op.so",
    srcs = ["duplicate_op.cc"],
)

tf_py_test(
    name = "duplicate_op_test",
    size = "small",
    srcs = ["duplicate_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    data = [":duplicate_op.so"],
)

tf_custom_op_library(
    name = "invalid_op.so",
    srcs = ["invalid_op.cc"],
)

tf_py_test(
    name = "invalid_op_test",
    size = "small",
    srcs = ["invalid_op_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
    data = [":invalid_op.so"],
)

tf_custom_op_library(
    name = "gpu_aware_record_input_op.so",
    srcs = ["gpu_aware_record_input_op.cc","gpu_record_yielder.h","gpu_record_yielder.cc"]
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
