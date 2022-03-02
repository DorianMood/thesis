import torch
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def process_batch_psnr(input1: torch.Tensor, input2: torch.Tensor):
    assert input1.shape == input2.shape
    N, C, H, W = input1.shape

    images1 = input1.cpu().detach().transpose(1, 3).numpy()
    images2 = input2.cpu().detach().transpose(1, 3).numpy()

    psnr_batch = []
    for i in range(N):
        # Calculate metric for each image
        psnr_batch.append(psnr(images1[i], images2[i]))
    psnr_batch = torch.tensor(psnr_batch, device=input1.device)

    return psnr_batch


def process_batch_ssim(input1: torch.Tensor, input2: torch.Tensor):
    assert input1.shape == input2.shape
    N, C, H, W = input1.shape

    images1 = input1.cpu().detach().transpose(1, 3).numpy()
    images2 = input2.cpu().detach().transpose(1, 3).numpy()

    ssim_batch = []
    for i in range(N):
        ssim_batch.append(ssim(images1[i], images2[i], multichannel=True))
    ssim_batch = torch.tensor(ssim_batch, device=input1.device)

    return ssim_batch

if __name__ == '__main__':
    image1, image2 = torch.rand(8, 3, 128, 128), torch.rand(8, 3, 128, 128)
    print(f'Random images SSIM = {process_batch_ssim(image1, image2).mean()}, PSNR = {process_batch_psnr(image1, image2).mean()}')
