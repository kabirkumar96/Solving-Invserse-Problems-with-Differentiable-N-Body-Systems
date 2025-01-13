import numpy as np
import pickle


def initValues_multi_planet(num_bodies=3):

    tau = 0.01

    with open('datasets/multi_planet_data.pkl', 'rb') as f:
        multi_planet_data = pickle.load(f)

    m = np.array(multi_planet_data['m'])
    r = np.array(multi_planet_data['r'])
    v = np.array(multi_planet_data['v'])
    n = np.array(multi_planet_data['n'])

    print(m.shape, r.shape, v.shape, n.shape)
    return tau, num_bodies, m[:num_bodies], r[:num_bodies], v[:num_bodies]


def initValues(num_bodies=3, equal_mass=False):
    tau = 0.01

    r = [[-7.673580434738827E-03, -3.302239590214931E-03, 2.073556671274906E-04],
         [3.452597940573748E-01, -1.498122010844959E-01, -4.413780254122306E-02],
         [1.031676785783527E-01, -7.220322696819025E-01, -1.605894448118791E-02]]

    if equal_mass:
        v = [[0, 0, 0],
             [5.305044905894823E-03, 2.724866176413712E-02,
              1.741157849018483E-03],
             [1.986026764570113E-02, 3.004963198838360E-03, -1.104347674861390E-03]]
        m = [1, 1, 1]
    else:
        v = [[5.284080958425666E-06, -6.612725729575814E-06, -5.777615059407366E-08],
             [5.305044905894823E-03, 2.724866176413712E-02, 1.741157849018483E-03],
             [1.986026764570113E-02, 3.004963198838360E-03, -1.104347674861390E-03]]

        m = [1, 3.302e23 / 1988500e24, 48.685e23 / 1988500e24]

    return tau, np.array(num_bodies), np.array(m[:num_bodies]), np.array(r[:num_bodies]), np.array(v[:num_bodies])
