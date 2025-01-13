import tensorflow as tf
import tf_keras
import TensorCode.kepler_solver as ks

def kepler_solver(vector):
    if tf.reduce_all(tf.equal(vector, 0)):
        return tf.zeros(6, dtype=tf.float64)
    else:
        tau = vector[0]
        mij = vector[1]
        r0 = vector[2:5]
        v0 = vector[5:8]
        keplerConstant = mij * (6.67418478 * 10 ** -11) * (24 * 60 * 60) ** 2 * 1988500 * 10 ** 24 * (
            1 / (1.496 * 10 ** 11) ** 3)  # In AU/M*d**2
        # keplerConstant = mij * ((4 * pi ** 2) / 365 ** 2)

        r0, v0 = ks.kepler_step(keplerConstant, tau, r0, v0)

        # Concatenate the results into a single vector
        return tf.concat([r0, v0], axis=0)
    

def convert_model_to_matrix_ops(model):
    """
    Convert a trained TensorFlow neural network to explicit matrix multiplication
    
    Args:
        model: Trained TensorFlow Sequential or Functional model
    
    Returns:
        A function that performs matrix multiplication equivalent to the model
    """
    # Extract weights and biases from each layer
    weights_and_biases = []
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            # Get weights and biases
            W = layer.get_weights()[0]  # Weight matrix
            b = layer.get_weights()[1]  # Bias vector
            
            # Get activation function
            activation = layer.activation
            
            # Store as tuple
            weights_and_biases.append((W, b, activation))
    
    def matrix_multiplication_forward(x):
        """
        Perform forward pass using explicit matrix multiplication
        
        Args:
            x: Input tensor
        
        Returns:
            Output after passing through the network
        """
        current = x
        
        # Iterate through layers
        for W, b, activation in weights_and_biases:
            # Perform matrix multiplication and add bias
            current = tf.matmul(current, W) + b
            
            # Apply activation function
            if activation != tf.keras.activations.linear:
                current = activation(current)
        
        return current
    
    return matrix_multiplication_forward

    
def kepler_solver_model(vector, model):
    if tf.reduce_all(tf.equal(vector, 0)):
        return tf.zeros(6, dtype=tf.float64)
    else:
        # print(vector)
        tau = vector[0]
        mij = vector[1]
        r0 = vector[2:5]
        v0 = vector[5:8]
        keplerConstant = mij * (6.67418478 * 10 ** -11) * (24 * 60 * 60) ** 2 * 1988500 * 10 ** 24 * (
                1 / (1.496 * 10 ** 11) ** 3)  # In AU/M*d**2

        if len(model) > 0:

            model_name, model_solver = model[0], model[1]

            if model_name == 'ANN':
                matrix_forward = convert_model_to_matrix_ops(model_solver)
                output = matrix_forward(tf.reshape(vector[1:], (1, -1)))[0]
            elif model_name in ['RF', 'DT', 'NLR']:
                tf_keras.backend.eval(vector)
                output = model_solver.predict(tf.reshape(vector[1:], (1, -1)))[0]

        else:
            r0, v0 = ks.kepler_step(keplerConstant, tau, r0, v0)
            output = tf.concat([r0, v0], axis=0)

        output = tf.cast(output, tf.float64)

        return tf.concat([output[:3], output[3:]], axis=0)

@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64)
])
def do_step(tau, n, m, r, v):
    tauDiv2 = tf.multiply(tau, 0.5)
    r = tf.add(r, tf.multiply(v, tauDiv2))
    r, v = evolve_HW(tau, n, m, r, v)
    r = tf.add(r, tf.multiply(v, tauDiv2))
    return r, v

def do_step_tfmodel(tau, n, m, r, v, model):
    tauDiv2 = tf.multiply(tau, 0.5)
    tauDiv2 = tf.cast(tauDiv2, dtype=tf.double)
    r = tf.add(r, tf.multiply(v, tauDiv2))
    r, v = evolve_HW_tfmodel(tau, n, m, r, v, model)
    r = tf.add(r, tf.multiply(v, tauDiv2))
    return r, v

@tf.function(input_signature=[
    tf.TensorSpec(shape=(), dtype=tf.float64),
    tf.TensorSpec(shape=(), dtype=tf.int32),
    tf.TensorSpec(shape=(None,), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
    tf.TensorSpec(shape=(None, 3), dtype=tf.float64)
])
def evolve_HW(tau, n, m, r, v):
    maskMatrix2D = 1 - tf.eye(n, dtype=tf.float64)
    maskMatrix3D = tf.expand_dims(maskMatrix2D, 2)

    m = tf.reshape(m, (n, 1))

    mij = m * maskMatrix2D + tf.transpose(m * maskMatrix2D)

    mij_with_1_on_diagonal_instead_of_0 = mij + tf.eye(n, dtype=tf.float64)

    mu = (m * maskMatrix2D) * (tf.transpose(m * maskMatrix2D)) / \
        mij_with_1_on_diagonal_instead_of_0

    r_expanded = tf.expand_dims(r, 1) * maskMatrix3D
    v_expanded = tf.expand_dims(v, 1) * maskMatrix3D

    rr0 = r_expanded - tf.transpose(r_expanded, perm=[1, 0, 2])
    vv0 = v_expanded - tf.transpose(v_expanded, perm=[1, 0, 2])

    r0 = rr0 - vv0 * tau * 0.5

    tau = tf.broadcast_to(tau, (n, n, 1))
    mij = tf.expand_dims(mij, 2)
    concatenated = tf.concat([tau, mij, r0, vv0], axis=2)

    lower_triangular_1_matrix = tf.expand_dims(
        tf.linalg.band_part(maskMatrix2D, -1, 0), 2)

    concatenated = concatenated * lower_triangular_1_matrix
    concatenated = tf.reshape(concatenated, (-1, 8))

    result = tf.map_fn(kepler_solver, concatenated,
                       fn_output_signature=tf.TensorSpec(shape=None, dtype=tf.float64))

    result = tf.reshape(result, (n, n, 6))
    r1 = result[:, :, :3]
    v1 = result[:, :, 3:]

    r1 = r1 + tf.transpose(-r1, perm=[1, 0, 2])
    v1 = v1 + tf.transpose(-v1, perm=[1, 0, 2])

    rr1 = r1 - (v1 * (tau * 0.5))

    mu = tf.reshape(mu, (n, n, 1))

    dmr = tf.reduce_sum(mu * (rr1 - rr0), 1)
    dmv = tf.reduce_sum(mu * (v1 - vv0), 1)

    r = r + tf.divide(dmr, m)
    v = v + tf.divide(dmv, m)

    return r, v


# @tf.function(input_signature=[
#     tf.TensorSpec(shape=(), dtype=tf.float64),
#     tf.TensorSpec(shape=(), dtype=tf.int32),
#     tf.TensorSpec(shape=(None,), dtype=tf.float64),
#     tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
#     tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
#     tf.TensorSpec(shape=None, dtype=None)
# ])

def evolve_HW_tfmodel(tau, n, m, r, v, model):
    # print("IN evolve")

    tau = tf.cast(tau, tf.float64)
    n = tf.cast(n, tf.int32)
    m = tf.cast(m, tf.float64)
    r = tf.cast(r, tf.float64)
    v = tf.cast(v, tf.float64)
    maskMatrix2D = 1 - tf.eye(n, dtype=tf.float64)
    maskMatrix3D = tf.expand_dims(maskMatrix2D, 2)

    m = tf.cast(tf.reshape(m, (n, 1)), tf.float64)

    mij = m * maskMatrix2D + tf.transpose(m * maskMatrix2D)

    mij_with_1_on_diagonal_instead_of_0 = mij + tf.eye(n, dtype=tf.float64)

    mu = (m * maskMatrix2D) * (tf.transpose(m * maskMatrix2D)) / mij_with_1_on_diagonal_instead_of_0

    r_expanded = tf.expand_dims(r, 1) * maskMatrix3D
    v_expanded = tf.expand_dims(v, 1) * maskMatrix3D

    rr0 = r_expanded - tf.transpose(r_expanded, perm=[1, 0, 2])
    vv0 = v_expanded - tf.transpose(v_expanded, perm=[1, 0, 2])

    r0 = rr0 - vv0 * tau * 0.5

    tau = tf.broadcast_to(tau, (n, n, 1))
    mij = tf.expand_dims(mij, 2)
    # mij = tf.cast(mij, dtype=tf.float64)
    concatenated = tf.concat([tau, mij, r0, vv0], axis=2)

    lower_triangular_1_matrix = tf.expand_dims(tf.linalg.band_part(maskMatrix2D, -1, 0), 2)

    concatenated = concatenated * lower_triangular_1_matrix
    concatenated = tf.reshape(concatenated, (-1, 8))

    # tf.print("Concatenated")
    # tf.print(concatenated.shape)

    result = tf.map_fn(lambda vector: kepler_solver_model(vector=vector, model=model), concatenated,
                       fn_output_signature=tf.TensorSpec(shape=None, dtype=tf.float64))
    
    result = tf.reshape(result, (n, n, 6))
    r1 = result[:, :, :3]
    v1 = result[:, :, 3:]

    r1 = r1 + tf.transpose(-r1, perm=[1, 0, 2])
    v1 = v1 + tf.transpose(-v1, perm=[1, 0, 2])

    rr1 = r1 - (v1 * (tau * 0.5))

    mu = tf.reshape(mu, (n, n, 1))

    dmr = tf.reduce_sum(mu * (rr1 - rr0), 1)
    dmv = tf.reduce_sum(mu * (v1 - vv0), 1)

    r = r + tf.divide(dmr, m)
    v = v + tf.divide(dmv, m)

    return r, v