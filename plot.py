import matplotlib.pyplot as plt


def load_tuples(filename: str):
    tuples = []

    with open(filename, 'r') as f:
        for line in f:
            p, value = line.split()
            new_tuple = float(p), float(value)
            tuples.append(new_tuple)

    return tuples


def plot_error_data(out_file: str):
    sfm_error = load_tuples('plot/error_sfm.dat')
    pcsa_error = load_tuples('plot/error_pcsa.dat')

    plt.plot([p for p, _ in sfm_error], [err for _, err in sfm_error])
    plt.plot([p for p, _ in pcsa_error], [err for _, err in pcsa_error])

    plt.legend(['SFM p=0.85', 'PCSA'])
    plt.xlabel('Tamaño de los índices (p)')
    plt.ylabel('Error Absoluto Medio')

    plt.savefig(out_file)
    plt.close()


def plot_time(out_file: str):
    sfm_time = load_tuples('plot/time_sfm.dat')
    pcsa_time = load_tuples('plot/time_pcsa.dat')

    plt.plot([p for p, _ in sfm_time], [err for _, err in sfm_time])
    plt.plot([p for p, _ in pcsa_time], [err for _, err in pcsa_time])
    plt.legend(['SFM p=0.85', 'PCSA'])
    plt.xlabel('Tamaño de los índices (p)')
    plt.ylabel('Tiempo (s)')
    plt.savefig(out_file)
    plt.close()


def plot_spur(out_file: str):
    sfm_spur = load_tuples('plot/mean_spur_sfm.dat')
    pcsa_spur = load_tuples('plot/mean_spur_pcsa.dat')

    plt.plot([p for p, _ in sfm_spur], [spur for _, spur in sfm_spur])
    plt.plot([p for p, _ in pcsa_spur], [spur for _, spur in pcsa_spur])
    plt.legend(['SFM p=0.85', 'PCSA'])
    plt.xlabel('Tamaño de los índices (p)')
    plt.ylabel('Número de intersecciones erróneas')
    plt.savefig(out_file)
    plt.close()


if __name__ == '__main__':
    plot_spur('plot/spur.png')
    plot_time('plot/time.png')
    plot_error_data('plot/error.png')
