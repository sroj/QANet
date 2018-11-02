import tensorflow_hub as hub

import tensorflow as tf

print(tf.__version__)

# elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
#
# tokens_input = [["the", "cat", "is", "on", "the", "mat"],
#                 ["dogs", "are", "in", "the", "fog", ""]]
#
# tokens_length = [6, 5]
#
# embeddings = elmo(
#     inputs={
#         "tokens": tokens_input,
#         "sequence_len": tokens_length
#     },
#     signature="tokens",
#     as_dict=True)["elmo"]


elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
embeddings = elmo(
    ["the cat is on the mat", "dogs are in the fog"],
    signature="default",
    as_dict=True)["elmo"]

print(embeddings)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    embeddings_numpy = sess.run(embeddings)

    print(embeddings_numpy.tolist())
