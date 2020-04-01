import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam


class LRMultiplierAdam(Adam):
    """Adam optimizer with varying per-layer learning rates.

    Parameters
    ----------
    multipliers : dict of layername (str) -> multipler (float)
    """

    def __init__(self, *args, multipliers={}, **kwargs):
        super(LRMultiplierAdam, self).__init__(*args, **kwargs)
        with K.name_scope(self.__class__.__name__):
            self.multipliers = {k: K.variable(v) for k, v in multipliers.items()}

    def get_updates(self, loss, params):
        # Mostly the same code as Adam class, with added multiplier variables.
        # Keras code from:
        # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/optimizers.py#L456
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (
                1.0 / (1.0 + self.decay * K.cast(self.iterations, K.dtype(self.decay)))
            )

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (
            K.sqrt(1.0 - K.pow(self.beta_2, t)) / (1.0 - K.pow(self.beta_1, t))
        )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            layername = p.name.split("/", 1)[0]
            mult = self.multipliers.get(layername, 1.0)

            m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1.0 - self.beta_2) * K.square(g)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - mult * lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - mult * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, "constraint", None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates
