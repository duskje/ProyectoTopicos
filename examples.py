from simulation import compare_pcsa_sfm
from plot import plot_time, plot_spur, plot_error_data


def compare_sfm_vs_pcsa():
    compare_pcsa_sfm(10)

    plot_spur('plot/spur.png')
    plot_time('plot/time.png')
    plot_error_data('plot/error.png')

if __name__ == '__main__':
    compare_sfm_vs_pcsa()