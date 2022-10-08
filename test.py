import time
import numpy as np
import matplotlib.pyplot as plt
from simplebrainviewer import plot_brain
from typing import List

from twixtools import read_twix, map_twix
from twixtools.mdb import Mdb
from pygrappa import mdgrappa


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
            while "PHASCOR" in scan_data.get_active_flags():
                increment += 1
                scan_data = mdb_list[idx + increment]
            # change partition index and line index
            mdb.mdh["sLC"]["ushLine"] = 0
            mdb.mdh["sLC"]["ushPartition"] = scan_data.cPar


# get ffts with shifts
def fft_w_shift(x, axis):
    return np.fft.ifftshift(np.fft.fft(np.fft.fftshift(x), axis=axis))


def ifft_w_shift(x, axis):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x), axis=axis))


def fft2_w_shift(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))


def ifft2_w_shift(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))


def rms_comb(sig, axis=3):
    return np.sqrt(np.sum(abs(sig) ** 2, axis))


# N/2 Nyquist ghost filter
# Why does this work? I have no idea...
# But took it from here:
# https://github.com/pehses/twixtools/blob/master/demo/recon_example.ipynb
def ghost_filter(data, phase_corr, add_partition=True):
    # get number of columns
    ncol = phase_corr.shape[0]

    # get ifft of phase correction
    phase_corr_ifft = ifft_w_shift(phase_corr, axis=0)

    # compute slope
    slope = np.angle(
        (np.conj(phase_corr_ifft[1:, ...]) * phase_corr_ifft[:-1, ...])
        .sum(axis=0, keepdims=True)
        .sum(axis=2, keepdims=True)
    )

    # get phase correction function
    x = np.arange(ncol) - ncol // 2
    pc_corr = np.exp(1j * slope * x[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis])

    # phase correct the signal
    sig = ifft_w_shift(data, axis=0)
    new_sig = sig * pc_corr
    new_sig = new_sig.sum(axis=-1)

    # if add partition, add another empty layer to the partition axis
    if add_partition:
        new_sig = np.append(new_sig, np.zeros((*new_sig.shape[:-1], 1)), axis=3)

    # return the data back to k-space
    return fft_w_shift(new_sig, axis=0)


if __name__ == "__main__":
    # read in dat file
    dat_file = read_twix(
        "/home/vanandrew/Data/a_ep_seg_fid_mdt_data/meas_MID00119_FID43267_a_ep_seg_fid_mdt.dat"
    )

    # fix phase correction indexing
    fix_phase_corr(dat_file[1]["mdb"])

    # map data to arrays
    mapped_data = map_twix(dat_file[1])

    # get data
    mapped_data["image"].flags["remove_os"] = True
    mapped_data["image"].flags["average"]["Seg"] = False
    img_data = lambda t: mapped_data["image"][:, :, :, :, :, :, :, t].squeeze().T
    num_frames = mapped_data["image"].shape[7]

    # get ref data
    mapped_data["refscan"].flags["remove_os"] = True
    mapped_data["refscan"].flags["average"]["Seg"] = False
    mapped_data["refscan"].flags["skip_empty_lead"] = True
    ref_data = lambda t: mapped_data["refscan"][:, :, :, :, :, :, :, t].squeeze().T
    num_ref_lines = mapped_data["refscan"].shape[-6]
    total_ref_lines = mapped_data["refscan"].base_size[-6]
    ref_line_start = total_ref_lines - num_ref_lines
    ref_line_end = total_ref_lines

    # get phase correction data
    mapped_data["phasecorr"].flags["remove_os"] = True
    mapped_data["phasecorr"].flags["average"]["Seg"] = False
    mapped_data["phasecorr"].flags["average"]["Ave"] = False
    phase_corr = lambda t: mapped_data["phasecorr"][:, :, :, :, :, :, :, t].squeeze().T

    # for each time point
    recon_data = list()
    for t in range(num_frames):
        print(f"Frame {t}")
        print("Filtering N/2 Nyquist Ghosts...")
        # filter for N/2 ghosting
        data = ghost_filter(img_data(t), phase_corr(t))
        ref = ghost_filter(
            ref_data(t),
            phase_corr(t)[:, :, :, ref_line_start:ref_line_end, :],
            add_partition=False,
        )
        print("Done.")

        # reorganize arrays
        data = np.moveaxis(data, [0, 2, 3, 1], [0, 1, 2, 3])
        ref = np.moveaxis(ref, [0, 2, 3, 1], [0, 1, 2, 3])

        # run grappa
        print("Running GRAPPA, please wait...")
        start = time.perf_counter()
        gdata = mdgrappa(data, ref)
        print(f"Time Elapsed: {time.perf_counter() - start}")

        # recon with fft, combining coils through sum of squares
        rdata = rms_comb(
            np.fft.fftshift(
                np.fft.ifftn(np.fft.ifftshift(gdata, axes=(0, 1, 2)), axes=(0, 1, 2)),
                axes=(0, 1, 2),
            )
        )
        recon_data.append(rdata)

    # stack frames
    recon_data = np.stack(recon_data, axis=3)
    np.save("recon_data.npy", recon_data)

    # show image
    # plot_brain(np.abs(recon_data))
