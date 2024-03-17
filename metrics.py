import os
import glob
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import numpy as np

from eval_utils import ssim, uint8_to_float32


def psnr(gt_image, gen_image):
    mse_fn = lambda x, y: np.mean((x - y)**2)
    return -10 * np.log10(mse_fn(gt_image, gen_image))


def plot_ray_termination_distribution(z_vals, weights, expname):
    plt.plot(z_vals, weights)
    plt.xlabel('Abscissa along ray')
    plt.ylabel(r'Ray termination probability $h$')
    # plt.title("Ray termination distribution")
    plt.savefig(f"figures/ray_termination_distribution_{expname}.png")
    plt.show()
    

def fetch_test_generations(experiment_name):
    """ 
    Returns:
        - an array of shape (n_test_evaluations, n_test_scenes, *rgb_image_shape) that contains the generated images for the test scenes.
        - the list of numbers of iterations at which the evaluations were performed.
    """
    experiment_path = os.path.join('logs', experiment_name, experiment_name)
    generated_scenes_over_training = []
    iterations = []
    for test_eval_file in glob.glob(os.path.join(experiment_path, "testset*")):
        iterations.append(int(test_eval_file.split("_")[-1]))
        generated_scenes = []
        for scene_file in glob.glob(os.path.join(test_eval_file, "*.npz")):
            generated_scenes.append(np.load(scene_file)["rgb"])
        generated_scenes_over_training.append(generated_scenes)
    return np.array(generated_scenes_over_training), np.array(iterations)


def fetch_test_ground_truth(dataset_name, test_scenes_ids, factor=4):
    """ Returns an array of shape (n_test_scenes, *rgb_image_shape) that contains the ground truth images for the test scenes."""
    data_path = os.path.join('data', dataset_name, f"images_{factor}")
    ground_truth_images = []
    scenes_paths = glob.glob(os.path.join(data_path, "*.png"))
    for scene_id in test_scenes_ids:
        ground_truth_images.append(uint8_to_float32(imageio.imread(scenes_paths[scene_id])))
    return np.array(ground_truth_images)


def plot_psnr_over_training(experiment_name, dataset_name, test_scenes_ids, factor=4):
    # Fetch the test generations and ground truth
    generated_scenes, iterations = fetch_test_generations(experiment_name)
    ground_truth_scenes = fetch_test_ground_truth(dataset_name, test_scenes_ids, factor=factor)

    # Compute test PSNR over training
    psnr_means = []
    psnr_mins = []
    psnr_maxs = []
    for eval_nb in range(generated_scenes.shape[0]):
        psnrs_curr_eval = [psnr(gt, gen) for gt, gen in zip(ground_truth_scenes, generated_scenes[eval_nb])]
        psnr_means.append(np.mean(psnrs_curr_eval))
        psnr_mins.append(np.min(psnrs_curr_eval))
        psnr_maxs.append(np.max(psnrs_curr_eval))

    # Plot the PSNR values
    plt.plot(iterations/1000, psnr_means, color="tab:blue", label='Mean PSNR')
    plt.fill_between(iterations/1000, np.array(psnr_mins), np.array(psnr_maxs), alpha=0.3, label='Min-Max PSNR interval')
    plt.xlabel('Iterations (in thousands)')
    plt.ylabel('Test PSNR')
    plt.title('Test PSNR over training')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def plot_psnr_over_training_with_reference(experiment_name, dataset_name, test_scenes_ids, reference_experiment_name=None, factor=4):
    # Fetch the test generations and ground truth
    generated_scenes, iterations = fetch_test_generations(experiment_name)
    reference_generated_scenes, reference_iterations = fetch_test_generations(reference_experiment_name)
    ground_truth_scenes = fetch_test_ground_truth(dataset_name, test_scenes_ids, factor=factor)
    
    max_iterations = min(iterations[-1], reference_iterations[-1])
    generated_scenes = generated_scenes[:np.searchsorted(iterations, max_iterations)+1]
    reference_generated_scenes = reference_generated_scenes[:np.searchsorted(reference_iterations, max_iterations)+1]
    iterations = iterations[:np.searchsorted(iterations, max_iterations)+1]
    reference_iterations = reference_iterations[:np.searchsorted(reference_iterations, max_iterations)+1]
    
    # Compute test PSNR over training
    psnr_means = []
    psnr_mins = []
    psnr_maxs = []
    psnr_ref_means = []
    psnr_ref_mins = []
    psnr_ref_maxs = []
    for eval_nb in range(generated_scenes.shape[0]):
        psnrs_curr_eval = [psnr(gt, gen) for gt, gen in zip(ground_truth_scenes, generated_scenes[eval_nb])]
        psnr_means.append(np.mean(psnrs_curr_eval))
        psnr_mins.append(np.min(psnrs_curr_eval))
        psnr_maxs.append(np.max(psnrs_curr_eval))
        psnrs_ref_curr_eval = [psnr(gt, gen) for gt, gen in zip(ground_truth_scenes, reference_generated_scenes[eval_nb])]
        psnr_ref_means.append(np.mean(psnrs_ref_curr_eval))
        psnr_ref_mins.append(np.min(psnrs_ref_curr_eval))
        psnr_ref_maxs.append(np.max(psnrs_ref_curr_eval))
    
    # Plot the PSNR values
    plt.plot(iterations/1000, psnr_means, color="tab:blue", label='Mean PSNR (KL loss)')
    plt.fill_between(iterations/1000, np.array(psnr_mins), np.array(psnr_maxs), alpha=0.3, label='Min-Max PSNR interval (KL loss)')
    plt.plot(reference_iterations/1000, psnr_ref_means, color="tab:red", linestyle="--", label='Mean PSNR (MSE loss)')
    plt.fill_between(reference_iterations/1000, np.array(psnr_ref_mins), np.array(psnr_ref_maxs), alpha=0.3, color="tab:red", label='Min-Max PSNR interval (MSE loss)')
    plt.xlabel('Iterations (in thousands)')
    plt.ylabel('Test PSNR')
    plt.title('Test PSNR over training')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    experiment_name = "table_5v_no_ds"
    dataset_name = "table_5v"
    # reference_experiment_name = "table_35v_no_ds"
    reference_experiment_name = None
    test_scenes_ids = [5, 6, 7, 8]
    plot_psnr_over_training(experiment_name, dataset_name, test_scenes_ids, factor=4)
    # plot_psnr_over_training_with_reference(experiment_name, dataset_name, test_scenes_ids, reference_experiment_name, factor=4)