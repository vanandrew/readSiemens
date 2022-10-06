import time
import numpy as np
import matplotlib.pyplot as plt
from simplebrainviewer import plot_brain
from typing import List

from mapvbvd import mapVBVD
from twixtools import read_twix, map_twix
from twixtools.mdb import Mdb
from pygrappa import mdgrappa


# make plot
def make_plot(orig: np.ndarray, filtered: np.ndarray, filtered2: np.ndarray) -> None:
    """Make plot of original and filtered data."""
    f = plt.figure()
    if filtered is not None:
        f.add_subplot(1, 3, 1)
    else:
        f.add_subplot(1, 2, 1)
    plt.imshow(np.abs(orig).T)
    plt.title("no filter", color="black")
    if filtered is not None:
        f.add_subplot(1, 3, 2)
        plt.imshow(np.abs(filtered).T)
        plt.title("filtered (roi)", color="black")
        f.add_subplot(1, 3, 3)
    else:
        f.add_subplot(1, 2, 2)
    plt.imshow(np.abs(filtered2).T)
    plt.title("filtered (cal)", color="black")
    plt.show()


# make filt from data
def make_filt(even: np.ndarray, odd: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Make and filter image."""
    # fft the data
    even_fft = np.fft.fftshift(np.fft.fft2(even))
    odd_fft = np.fft.fftshift(np.fft.fft2(odd))

    # get indices where even_fft is zero to avoid divide by zero
    zero_indices = np.where(even_fft == 0)
    divider = even_fft.copy()
    divider[zero_indices] = 1

    # compute filter
    filt = odd_fft * np.conj(even_fft) / np.abs(divider) ** 2

    # average the roi and return
    return np.mean(filt[:, roi], axis=1)[:, np.newaxis]


# apply calibration filt
def make_cal_filt(even_c: np.ndarray, odd_c: np.ndarray) -> np.ndarray:
    """Make and apply calibration filter."""
    # make filt
    filt = (odd_c / even_c)[:, np.newaxis]

    # return filt
    return filt


# def N/2 ghost filter
def ghost_filter(
    data: np.ndarray,
    phase_corr: np.ndarray,
    roi_slice: int = 30,
    add_partition: bool = True,
    plot: bool = False,
    increment: int = 2,
) -> np.ndarray:
    # store filtered partitions
    shape = list(data.shape[0:-1])
    if add_partition:  # only if add partition is true
        shape[-1] += 1
    shape = tuple(shape)
    filtered_data = np.zeros(shape).astype(np.complex64)

    # for each channel
    for c in range(data.shape[1]):
        print(f"Channel: {c}")

        # # make filter from roi method
        # filt_roi = make_filt(
        #     data[:, c, :, roi_slice, 0], data[:, c, :, roi_slice, 1], np.r_[0:10, 80:90]
        # )

        # for each partition
        for p in range(0, data.shape[3], increment):
            print(f"Filtering Partition: {p}", end="\r")
            # get ffts of calibration data
            s_prime_even_calib_fft = np.fft.fftshift(np.fft.fft(phase_corr[:, c, p, 0]))
            s_prime_odd_calib_fft = np.fft.fftshift(np.fft.fft(phase_corr[:, c, p, 1]))

            # make filter from calibration method
            filt_cal = make_cal_filt(s_prime_even_calib_fft, s_prime_odd_calib_fft)

            # get even/odd readout slices
            s_prime_even = data[:, c, :, p, 0]
            s_prime_odd = data[:, c, :, p, 1]

            # create image with no filtering for comparison
            s_prime = s_prime_even + s_prime_odd
            s_prime_fft = np.fft.fftshift(np.fft.fft2(s_prime))

            # get ffts of even/odd readout slices
            s_prime_even_fft = np.fft.fftshift(np.fft.fft2(s_prime_even))
            s_prime_odd_fft = np.fft.fftshift(np.fft.fft2(s_prime_odd))

            # apply filter
            # s_prime_r_fft_roi = np.fft.fftshift(
            #     s_prime_odd_fft + s_prime_even_fft * filt_roi, axes=1
            # )
            s_prime_r_fft_roi = None
            s_prime_r_fft_cal = s_prime_odd_fft + s_prime_even_fft * filt_cal
            filtered_data[:, c, :, p] = np.fft.ifft2(
                np.fft.ifftshift(s_prime_r_fft_cal)
            )

            # plot original and filtered image
            if plot:
                make_plot(s_prime_fft, s_prime_r_fft_roi, s_prime_r_fft_cal)
        print("Filtering Partition: Done")

    # return filtered data
    return filtered_data


# fix phase correction data
# phase correction collected at center of k-space
# so we need to change the indexing so it corresponds to where the
# data should be indexed, which is the location of partiion index of the scan
# preceding after the phase correction
def fix_phase_corr(mdb_list: List[Mdb]) -> None:
    for idx, mdb in enumerate(mdb_list):
        active_flags = mdb.get_active_flags()
        if "PHASCOR" in active_flags:
            # get first scan in data slice
            increment = 1
            scan_data = mdb_list[idx + increment]
            while("PHASCOR" in scan_data.get_active_flags()):
                increment += 1
                scan_data = mdb_list[idx + increment]
            # change partition index and line index
            mdb.mdh['sLC']['ushLine'] = 0
            mdb.mdh['sLC']['ushPartition'] = scan_data.cPar


if __name__ == "__main__":
    # read in dat file
    # dat = mapVBVD(
    #     "/home/vanandrew/Data/a_ep_seg_fid_mdt_data/meas_MID00119_FID43267_a_ep_seg_fid_mdt.dat"
    # )
    dat_file = read_twix("/home/vanandrew/Data/a_ep_seg_fid_mdt_data/meas_MID00119_FID43267_a_ep_seg_fid_mdt.dat")

    # fix phase correction indexing
    fix_phase_corr(dat_file[1]['mdb'])

    # map data to arrays
    mapped_data = map_twix(dat_file[1])

    # get data
    mapped_data["image"].flags["remove_os"] = True
    mapped_data["image"].flags["average"]["Seg"] = False
    img_data = mapped_data["image"][:].squeeze().T

    # get ref data
    mapped_data["refscan"].flags["remove_os"] = True
    mapped_data["refscan"].flags["average"]["Seg"] = False
    mapped_data["refscan"].flags["skip_empty_lead"] = True
    ref_data = mapped_data["refscan"][:].squeeze().T

    # get phase correction data
    mapped_data["phasecorr"].flags["remove_os"] = True
    mapped_data["phasecorr"].flags["average"]["Seg"] = False
    phase_corr = mapped_data["phasecorr"][:].squeeze().T

    # get data
    # img_data = dat[1].image[""].squeeze()
    # img_data = np.load("img_data.npy")
    # ref_data = dat[1].refscan[""].squeeze()
    # ref_data = np.load("ref_data.npy")

    # filter for N/2 ghosting
    # for each time point
    data = list()
    ref = list()
    for t in range(img_data.shape[4]):
        print(f"Frame {t}")
        data.append(ghost_filter(img_data[:, :, :, :, t, :], phase_corr[:, :, :, t, :], plot=True))
        ref.append(
            ghost_filter(
                ref_data[:, :, :, :, t, :],
                phase_corr[:, :, :, t, :],
                roi_slice=6,
                add_partition=False,
                increment=1,
            )
        )
    # stack frames
    data = np.stack(data, axis=4)
    ref = np.stack(ref, axis=4)

    # reoganize arrays
    data = np.moveaxis(data, [0, 2, 3, 1, 4], [0, 1, 2, 3, 4])
    ref = np.moveaxis(ref, [0, 2, 3, 1, 4], [0, 1, 2, 3, 4])
    print(data.shape)

    # recon data
    # recon_data = list()
    recon_data = np.load("recon_data.npy")

    # # for each time point
    # for t in range(data.shape[4]):
    #     print(f"Recon Frame: {t}")
    #     print("Running GRAPPA, please wait...")
    #     start = time.perf_counter()
    #     # run grappa
    #     gdata = mdgrappa(
    #         data[:, :, :, :, t],
    #         ref[:, :, :, :, t],
    #     )
    #     print(f"Time Elapsed: {time.perf_counter() - start}")

    #     # recon with fft, combining coils through sum of squares
    #     rdata = np.sqrt(
    #         np.sum(
    #             np.fft.fftshift(np.fft.fftn(gdata, axes=(0, 1, 2)), axes=(0, 1, 2))
    #             ** 2,
    #             axis=3,
    #         )
    #     )
    #     recon_data.append(rdata)

    # stack frames
    # recon_data = np.stack(recon_data, axis=3)
    # np.save("recon_data.npy", recon_data)

    # show image
    plot_brain(np.abs(recon_data))
