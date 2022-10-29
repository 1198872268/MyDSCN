

def regularization1(self, reuse=False):
    combined = []
    for i in range(self.n_class):
        ui = self.Us[i]
        uiT = ui.t()
        temp_sum = []
        for j in range(self.n_class):
            if j == i:
                continue
            uj = self.Us[j]
            s = tf.reduce_sum((tf.matmul(uiT, uj)) ** 2)
            temp_sum.append(s)
        combined.append(tf.add_n(temp_sum))
    return tf.add_n(combined) / self.n_class


def regularization2(self, reuse=False):
    combined = []
    for i in range(self.n_class):
        ui = self.Us[i]
        uiT = tf.transpose(ui)
        s = tf.reduce_sum((tf.matmul(uiT, ui) - tf.eye(self.rank)) ** 2)
        combined.append(s)
    return tf.add_n(combined) / self.n_class