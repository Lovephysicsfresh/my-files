"""第4步：严格按 Keras MPNN 官方示例 + 作者 checkpoint 重建网络，并做一次前向。

关键点（现在已完全确定，不再靠猜）：

作者的 message passing 就是 Keras 官方示例
https://keras.io/examples/graph/mpnn-molecular-graphs/ 的 EdgeNetwork + GRUCell：

    atom_features 先零填充到 64 维            (4 -> 64，pad 60)
    每一步：
        bond = matmul(edge_features, W) + b    # W: (3, 4096)  <- checkpoint 里就是 [3,4096]
        bond = reshape(bond, (-1, 64, 64))
        neigh = gather(h, pair_indices[:,1])   # (E,64)
        t = matmul(bond, neigh[...,None])      # (E,64)
        agg = unsorted_segment_sum(t, pair_indices[:,0], N)
        h = GRUCell(64)(agg, h)

4096 = 64 x 64 来自"先把节点特征填到 64 维"，所以 EdgeNetwork 的 atom_dim 就是 64。

读出层 = TransformerEncoderReadout：
    MultiHeadAttention(1 head, key_dim=64)
    LayerNorm(x + attn)
    LayerNorm(x + Dense(128,relu) -> Dense(64))
    GlobalAveragePooling1D
然后 Dense(128) -> 与三路分支拼接(272) -> Dense(512) -> Dense(256) -> Dense(1)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from step_03_build_graph import (
    GRAPH_DIR,
    build_conect_variant,
    build_dense_variant,
    node_features,
    parse_pdb,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT = PROJECT_ROOT / "OER2022" / "initial_train" / "checkpoint" / "OO_00_full"

EMBED_DIM = 64
MESSAGE_STEPS = 4
DENSE_DIM = 128


class Net:
    def __init__(self, ckpt: Path):
        self.reader = tf.train.load_checkpoint(str(ckpt))
        self.shapes = self.reader.get_variable_to_shape_map()

        self.edge_dense = keras.layers.Dense(EMBED_DIM * EMBED_DIM, name="message_step")
        self.gru_cell = keras.layers.GRUCell(EMBED_DIM, name="update_step")
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=1, key_dim=EMBED_DIM, name="attention"
        )
        self.ffn_0 = keras.layers.Dense(DENSE_DIM, activation="relu", name="ffn0")
        self.ffn_1 = keras.layers.Dense(EMBED_DIM, name="ffn1")
        self.ln1 = keras.layers.LayerNormalization(name="ln1")
        self.ln2 = keras.layers.LayerNormalization(name="ln2")

        self.d_graph = keras.layers.Dense(128, name="graph")
        self.d_int = keras.layers.Dense(64, name="intensity")
        self.d_pos = keras.layers.Dense(64, name="position")
        self.d_tsne = keras.layers.Dense(16, name="tsne")
        self.d512 = keras.layers.Dense(512, name="head1")
        self.d256 = keras.layers.Dense(256, name="head2")
        self.d1 = keras.layers.Dense(1, name="out")

    def _v(self, name):
        return self.reader.get_tensor(name)

    def load(self):
        self.edge_dense(tf.zeros((1, 3)))
        self.gru_cell(tf.zeros((1, EMBED_DIM)), tf.zeros((1, EMBED_DIM)))
        self.attention(tf.zeros((1, 336, EMBED_DIM)), tf.zeros((1, 336, EMBED_DIM)))
        self.ffn_0(tf.zeros((1, EMBED_DIM)))
        self.ffn_1(tf.zeros((1, DENSE_DIM)))
        self.ln1(tf.zeros((1, EMBED_DIM)))
        self.ln2(tf.zeros((1, EMBED_DIM)))
        self.d_graph(tf.zeros((1, EMBED_DIM)))
        self.d_int(tf.zeros((1, 7)))
        self.d_pos(tf.zeros((1, 7)))
        self.d_tsne(tf.zeros((1, 2)))
        self.d512(tf.zeros((1, 272)))
        self.d256(tf.zeros((1, 512)))
        self.d1(tf.zeros((1, 256)))

        p = "layer_with_weights-0/"
        self.edge_dense.set_weights([
            self._v(p + "message_step/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "message_step/bias/.ATTRIBUTES/VARIABLE_VALUE"),
        ])
        self.gru_cell.set_weights([
            self._v(p + "update_step/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "update_step/recurrent_kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "update_step/bias/.ATTRIBUTES/VARIABLE_VALUE"),
        ])

        p = "layer_with_weights-1/"
        self.attention.set_weights([
            self._v(p + "attention/_query_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "attention/_query_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "attention/_key_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "attention/_key_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "attention/_value_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "attention/_value_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "attention/_output_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "attention/_output_dense/bias/.ATTRIBUTES/VARIABLE_VALUE"),
        ])
        self.ffn_0.set_weights([
            self._v(p + "dense_proj/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "dense_proj/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE"),
        ])
        self.ffn_1.set_weights([
            self._v(p + "dense_proj/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "dense_proj/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE"),
        ])
        self.ln1.set_weights([
            self._v(p + "layernorm_1/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "layernorm_1/beta/.ATTRIBUTES/VARIABLE_VALUE"),
        ])
        self.ln2.set_weights([
            self._v(p + "layernorm_2/gamma/.ATTRIBUTES/VARIABLE_VALUE"),
            self._v(p + "layernorm_2/beta/.ATTRIBUTES/VARIABLE_VALUE"),
        ])

        for i, layer in enumerate(
            [self.d_graph, self.d_int, self.d_pos, self.d_tsne,
             self.d512, self.d256, self.d1], start=2
        ):
            p = f"layer_with_weights-{i}/"
            layer.set_weights([
                self._v(p + "kernel/.ATTRIBUTES/VARIABLE_VALUE"),
                self._v(p + "bias/.ATTRIBUTES/VARIABLE_VALUE"),
            ])
        return self

    def message_passing(self, h, pair_indices, edge_features):
        h = tf.pad(h, [(0, 0), (0, EMBED_DIM - tf.shape(h)[1])])
        for _ in range(MESSAGE_STEPS):
            bond = self.edge_dense(edge_features)                 # (E, 4096)
            bond = tf.reshape(bond, (-1, EMBED_DIM, EMBED_DIM))   # (E, 64, 64)
            neigh = tf.gather(h, pair_indices[1])                 # (E, 64)
            t = tf.matmul(bond, neigh[..., None])[..., 0]         # (E, 64)
            agg = tf.math.unsorted_segment_sum(
                t, pair_indices[0], num_segments=tf.shape(h)[0]
            )
            h, _ = self.gru_cell(agg, h)
        return h

    def readout(self, h):
        attn = self.attention(h[None], h[None], h[None])[0]
        x = self.ln1(h + attn)
        x = self.ln2(x + self.ffn_1(self.ffn_0(x)))
        return tf.reduce_mean(x, axis=0)                          # average pooling

    def predict(self, nodes, pair_indices, edge_features, intensity, position, tsne):
        h = self.message_passing(tf.constant(nodes, tf.float32), pair_indices, edge_features)
        g = self.d_graph(self.readout(h)[None])[0]
        u = tf.concat([
            g,
            self.d_int(tf.constant(intensity[None], tf.float32))[0],
            self.d_pos(tf.constant(position[None], tf.float32))[0],
            self.d_tsne(tf.constant(tsne[None], tf.float32))[0],
        ], axis=0)
        x = self.d512(u[None])[0]
        x = self.d256(x[None])[0]
        return float(self.d1(x[None])[0, 0])


def main():
    net = Net(CKPT).load()
    print("权重载入完成，checkpoint 变量数:", len(net.shapes))

    df = pd.read_csv(PROJECT_ROOT / "OER2022" / "initial_train" / "OER_train_initial.csv")
    res = pd.read_csv(
        PROJECT_ROOT / "OER2022" / "initial_train" / "result"
        / "OO_corr=0.882_RMSE=0.033_MAE=0.028_MAPE=5.512_MPE=3.245.csv"
    )

    atoms, conect = parse_pdb(GRAPH_DIR / "P01_00.pdb")
    nodes = node_features(atoms)

    for label, row_id in [("P01", "P01")]:
        row = df[df["id"] == row_id].iloc[0]
        intensity = row[[f"intensity_{i}" for i in range(7)]].to_numpy(np.float32)
        position = row[[f"position_{i}" for i in range(7)]].to_numpy(np.float32)
        tsne = row[["tsne_0", "tsne_1"]].to_numpy(np.float32)

        for name, builder in [("方案A 1296条真实键", build_conect_variant),
                              ("方案B 50400条全组合", build_dense_variant)]:
            pairs, edges = builder(atoms, conect)
            pred = net.predict(
                nodes,
                tf.constant(pairs, tf.int32),
                tf.constant(edges, tf.float32),
                intensity, position, tsne,
            )
            print(f"{label} {name}: pred = {pred:.6f}")

    r = res[res["id"] == "P01"].iloc[0]
    print("\n作者 P01: true =", r["true_avg"], " pred_avg =", r["pred_avg"])
    print("作者的 10 个预测值:", [round(r[str(i)], 6) for i in range(10)])


if __name__ == "__main__":
    main()
