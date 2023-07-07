import tensorflow as tf
import keras.api._v2.keras as keras


class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images, labels):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, labels, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_images, labels = data

        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator.
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                fake_images = self.generator(
                    random_latent_vectors, labels, training=True
                )

                fake_logits = self.discriminator(fake_images, labels, training=True)
                real_logits = self.discriminator(real_images, labels, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)

                # Add the gradient penalty to the original discriminator loss
                gp = self.gradient_penalty(batch_size, real_images, fake_images, labels)
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(
                random_latent_vectors, labels, training=True
            )

            gen_img_logits = self.discriminator(generated_images, labels, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


class Discriminator(keras.Model):
    def __init__(self, shape, label_dim, **kwargs):
        super().__init__()
        self.shape = shape
        self.label_dim = label_dim
        self.label_embedding = self.label_embedding_model()
        self.model = self.build_model()

    def build_model(self):
        input_shape = (self.shape[0], self.shape[1], self.shape[2] + 1)
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(256, kernel_size=4, strides=2, padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(512, kernel_size=4, strides=2, padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Flatten(),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1),
            ],
            name="discriminator",
        )
        return model

    def label_embedding_model(self):
        model = keras.Sequential(
            [
                keras.Input(shape=(self.label_dim,)),
                keras.layers.Dense(self.shape[0] * self.shape[1]),
                keras.layers.Reshape((self.shape[0], self.shape[1], 1)),
            ]
        )
        return model

    def call(self, data, labels):
        label_embedding = self.label_embedding(labels)
        inputs = tf.concat([data, label_embedding], axis=-1)
        return self.model(inputs)


class Generator(keras.Model):
    def __init__(self, latent_dim, label_dim, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential(
            [
                keras.Input(shape=(self.latent_dim + self.label_dim,)),
                keras.layers.Dense(2 * 2 * 256),
                keras.layers.BatchNormalization(),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Reshape((2, 2, 256)),
                keras.layers.UpSampling2D(interpolation="bilinear"),
                keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.UpSampling2D(interpolation="bilinear"),
                keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same"),
                keras.layers.BatchNormalization(),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.UpSampling2D(interpolation="bilinear"),
                keras.layers.Conv2D(64, kernel_size=3, strides=1),
                keras.layers.BatchNormalization(),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.UpSampling2D(interpolation="bilinear"),
                keras.layers.Conv2D(
                    3, kernel_size=3, strides=1, padding="same", activation="sigmoid"
                ),
            ],
            name="generator",
        )
        return model

    def call(self, data, labels):
        inputs = tf.concat([data, labels], axis=1)
        return self.model(inputs)
