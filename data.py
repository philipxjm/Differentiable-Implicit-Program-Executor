import numpy as np
import glob
import hyper_params as hp
# np.set_printoptions(threshold=sys.maxsize)


def read_sdf(filepath):
    with open(filepath) as fp:
        line = fp.readline()
        shape = [int(x) for x in line.split()]
        sign_matrix = np.zeros(shape)
        origin = [float(x) for x in fp.readline().split()]
        dx = float(fp.readline())
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    val = float(fp.readline())
                    sign_matrix[x][y][z] = 1 if val >= 0 else 0
        return sign_matrix, origin, dx


def read_sdf_dir(dir_path):
    sdf_names = sorted(glob.glob(dir_path + "*.sdf"))
    return sdf_names
    sdfs = []
    for i in range(len(sdf_names)):
        sdfs.append(pad_sign_matrix(read_sdf(sdf_names[i])[0], hp.GRID_SIZE))
    return np.array(sdfs)


def pad_sign_matrix(sign_matrix, n):
    padded_sign_matrix = np.ones((n, n, n))
    x_offset = (n - sign_matrix.shape[0]) // 2
    y_offset = (n - sign_matrix.shape[1]) // 2
    z_offset = (n - sign_matrix.shape[2]) // 2
    padded_sign_matrix[x_offset:sign_matrix.shape[0] + x_offset,
                       y_offset:sign_matrix.shape[1] + y_offset,
                       z_offset:sign_matrix.shape[2] + z_offset] = sign_matrix
    return padded_sign_matrix


def read_program(filepath):
    with open(filepath) as fp:
        line = fp.readline()
        program = [int(x) for x in line.split()]
        return np.array(program)


def read_program_dir(dir_path):
    program_names = sorted(glob.glob(dir_path + "*.txt"))
    return program_names
    programs = []
    for i in range(len(program_names)):
        programs.append(read_program(program_names[i]))
    return np.array(programs)


def build_vocab(program_names):
    vocabs = []
    for name in program_names:
        program = read_program(name)
        for token in program:
            vocabs.append(token)
    vocabs = set(vocabs)
    idx2token = {i: w for i, w in enumerate(vocabs)}
    token2idx = {w: i for i, w in enumerate(vocabs)}
    return idx2token, token2idx


def tokenize(program, token2idx):
    return np.array([token2idx[w] for w in program])


def generate_data(sdf_dir_path, prg_dir_path):
    sdf_names = sorted(glob.glob(sdf_dir_path + "*.sdf"))
    program_names = sorted(glob.glob(prg_dir_path + "*.txt"))
    idx2token, token2idx = build_vocab(program_names)
    print(len(token2idx))
    indices = np.random.permutation(len(program_names))
    for i in range(len(indices)):
        padded_sign_matrix = pad_sign_matrix(
            read_sdf(sdf_names[i])[0], hp.GRID_SIZE
        )
        program = tokenize(read_program(program_names[i]), token2idx)
        shape = padded_sign_matrix.shape
        programs = []
        positions = []
        labels = []
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    programs.append(program)
                    positions.append([x, y, z])
                    labels.append(
                        padded_sign_matrix[x][y][z]
                    )
        yield np.array(programs), np.array(positions), np.array(labels)[..., np.newaxis]


# print(read_sdf_dir("data/sdf_small/").shape)
# sign_matrix, origin, dx = read_sdf("data/sdf_small/1.sdf")
# print(pad_sign_matrix(sign_matrix, 32))
# print(np.any(pad_sign_matrix(sign_matrix, 32) == 0))
