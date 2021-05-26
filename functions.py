import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, StandardScaler
import gudhi as gd


def process_repeated_measures(df):
    """Function to process repeated measures data for each participant"""
    df = df.drop('subjectid', axis=1)
    df = df.transpose()
    df.columns = ['value']
    df['variable'] = df.index
    df['week'] = df['variable'].str.extract(r'(\d+)$')
    df['measure'] = df['variable'].str.replace('\d+$', '')
    df.drop(['variable'],
            axis=1,
            inplace=True)
    # Reshape LONG --> WIDE
    df = df.pivot(index='week',
                  values='value',
                  columns='measure')
    # LOCF
    df = df.fillna(method='ffill')
    # Fill missing values with 0        [!!! CHECK THIS IS A GOOD IDEA]
    df = df.fillna(0)
    # Scale
    scaled = pd.DataFrame(scale(df))
    scaled.columns = df.columns
    return(scaled)


def summarise_alpha_complex(simplex_tree):
    result_str = 'Alpha complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
        repr(simplex_tree.num_simplices()) + ' simplices - ' + \
        repr(simplex_tree.num_vertices()) + ' vertices.'
    print(result_str)
    fmt = '%s -> %.2f'
    for filtered_value in simplex_tree.get_filtration():
        print(fmt % tuple(filtered_value))


def landscapes_approx(diag_dim, x_min, x_max, nb_steps, nb_landscapes):
    landscape = np.zeros((nb_landscapes, nb_steps))
    step = (x_max - x_min) / nb_steps
    for i in range(nb_steps):
        x = x_min + i * step
        event_list = []
        for pair in diag_dim:
            b = pair[0]
            d = pair[1]
            if (b <= x) and (x <= d):
                if x >= (d+b)/2:
                    event_list.append((d-x))
                else:
                    event_list.append((x-b))
        event_list.sort(reverse=True)
        event_list = np.asarray(event_list)
        for j in range(nb_landscapes):
            if(j < len(event_list)):
                landscape[j, i] = event_list[j]
    return landscape


def make_landscape(df, mas=50, xm=20):
    # Alpha complex -----------------------------------------------------------
    ac = gd.AlphaComplex(points=df.values)
    # Simplex tree, persistence -----------------------------------------------
    simplex_tree = ac.create_simplex_tree(max_alpha_square=mas)
    pers = simplex_tree.persistence()
    # Discretize landscapes ---------------------------------------------------
    # First dimension
    D1 = landscapes_approx(simplex_tree.persistence_intervals_in_dimension(0),
                           x_min=0, x_max=xm, nb_steps=1000, nb_landscapes=3)
    # Second dimension
    D2 = landscapes_approx(simplex_tree.persistence_intervals_in_dimension(1),
                           x_min=0, x_max=xm, nb_steps=1000, nb_landscapes=3)
    L1, L2, L3 = D1
    L4, L5, L6 = D2
    # Combine
    L = np.concatenate([L1, L2, L3, L4, L5, L6])
    return({'land': L,
            'pers': pers})


def combine_landscapes(ss):
    alls = pd.DataFrame(np.vstack([v['land'] for k, v in ss.items()]))
    alls['id'] = list(ss.keys())
    alls.set_index('id', inplace=True)
    labels = [str(a) + '_' + str(b).zfill(3)
              for a in ['D1L1', 'D1L2', 'D1L3',
                        'D2L1', 'D2L2', 'D2L3']
              for b in np.arange(0, 1000)]
    alls.columns = labels
    return(alls)
