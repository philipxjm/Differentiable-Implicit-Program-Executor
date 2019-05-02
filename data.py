import numpy as np
import sys
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
    programs = []
    for i in range(len(program_names)):
        programs.append(read_program(program_names[i]))
    return np.array(programs)


def generate_data(padded_sign_matrices, programs):
    training_set = {
        "programs": [],
        "positions": [],
        "labels": []
    }
    testing_set = {
        "programs": [],
        "positions": [],
        "labels": []
    }
    indices = np.random.permutation(programs.shape[0])
    training_idx = indices[:int(programs.shape[0]*.8)]
    test_idx = indices[int(programs.shape[0]*.8):]
    shape = padded_sign_matrices.shape
    for i in range(len(training_idx)):
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    training_set["programs"].append(programs[training_idx[i]])
                    training_set["positions"].append([x, y, z])
                    training_set["labels"].append(
                        padded_sign_matrices[x][y][z]
                    )
    print("done with train")
    for i in range(len(test_idx)):
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    testing_set["programs"].append(programs[test_idx[i]])
                    testing_set["positions"].append([x, y, z])
                    testing_set["labels"].append(padded_sign_matrices[x][y][z])
    return training_set, testing_set


# print(read_sdf_dir("data/sdf_small/").shape)
# sign_matrix, origin, dx = read_sdf("data/sdf_small/1.sdf")
# print(pad_sign_matrix(sign_matrix, 32))
# print(np.any(pad_sign_matrix(sign_matrix, 32) == 0))
