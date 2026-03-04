import pandas as pd
import matplotlib as mpl 
#mpl.use("TkAgg")
#mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

def get_ue_colors(ue_list, cmap_name='tab10'):
    """
    ue_list: sıralı UE id’leri
    geri: { ue_id: RGBA tuple }
    """
    cmap = plt.get_cmap(cmap_name, len(ue_list))
    return {uid: cmap(i) for i, uid in enumerate(ue_list)}

def hist_to_df(bs_id, ue_idx,
               se_la_hist, se_sh_hist,
               sinr_hist, harq_hist,
               nbits_hist, ut_sched_hist):

    ns, _, nu          = se_la_hist.shape
    _, _, n_sym, n_sc  = ut_sched_hist.shape
    slot_recs, alloc_recs = [], []

    # slot‐bazlı kayıtlar
    for col, uid in enumerate(ue_idx):
        for s in range(ns):
            se_la = se_la_hist[s,0,col]
            se_sh = se_sh_hist[s,0,col]
            sinr  = sinr_hist[s,0,col]
            harq  = harq_hist[s,0,col]
            bits  = nbits_hist[s,0,col]
            slot_recs.append({
                "slot": s, "bs": bs_id, "ue": int(uid),
                "se_la":  np.nan if np.isnan(se_la) else float(se_la),
                "se_sh":  np.nan if np.isnan(se_sh) else float(se_sh),
                "sinr_db":np.nan if np.isnan(sinr) else float(sinr),
                "harq_ok":np.nan if np.isnan(harq) else int(harq),
                "bits":   np.nan if bits==0      else int(bits)
            })

    # RE‐tahsis kayıtları
    for s in range(ns):
        for l in range(n_sym):
            for k in range(n_sc):
                ue_id = ut_sched_hist[s,0,l,k]
                alloc_recs.append({
                    "slot": s, "sym": l, "sc": k,
                    "bs": bs_id,
                    "ue":  int(ue_id) if ue_id!=-1 else np.nan
                })

    return pd.DataFrame.from_records(slot_recs), \
           pd.DataFrame.from_records(alloc_recs)


def plot_timeseries_from_df(df, ue_list=None, title=""):
    """
    Zaman serisinde:
     - SINR ve LA‐SE kendi UE renginde
     - Shannon‐SE sabit siyah “x” ile
     - TBLER ve grid aynı
    """
    if ue_list is None:
        ue_list = sorted(df.ue.dropna().unique())
    ue_colors = get_ue_colors(ue_list, 'plasma')

    n_ue = len(ue_list)
    fig, axs = plt.subplots(3, n_ue,
                            figsize=(4*n_ue, 9),
                            sharex='col', sharey='row')
    if n_ue == 1: axs = axs.reshape(3,1)

    for j, uid in enumerate(ue_list):
        d = df[df.ue == uid]
        c = ue_colors[uid]

        # SINR
        axs[0,j].plot(d.slot, d.sinr_db, 'o-', color=c)
        axs[0,j].set_ylabel('Effective SINR [dB]')

        # SE: LA (renkli) vs Shannon (siyah)
        axs[1,j].plot(d.slot, d.se_la, 'o-',   color=c, label='OLLA')
        axs[1,j].plot(d.slot, d.se_sh, 'x--',  color='k', label='Shannon')
        axs[1,j].set_ylabel('Spectral eff. [bps/Hz]')
        axs[1,j].legend()

        # TBLER
        #ok        = d.harq_ok == 1
        #cum_bler  = 1 - ok.cumsum() / ((~d.harq_ok.isna()).cumsum())

        ok              = d.harq_ok == 1
        sched_mask      = ~d.harq_ok.isna()          # planlanan slotlar True
        cum_bler        = 1 - ok.cumsum() / sched_mask.cumsum()
        cum_bler[~sched_mask] = np.nan               # planlanmayan slotları bastır
        axs[2,j].plot(d.slot, cum_bler, 'o-', color=c)

        axs[2,j].plot(d.slot, cum_bler, 'o-', color=c)
        axs[2,j].axhline(0.1, ls='--', color='k')
        axs[2,j].set_ylabel('TBLER')

        axs[0,j].set_title(f'UE {uid}')

    for ax in axs.flat:
        ax.grid()
        ax.set_xlabel('Slot')

    fig.suptitle(title)
    fig.tight_layout()

def plot_allocation_from_df(alloc_df, num_symbols, num_subcarriers, title=""):
    """
    Tüm slot×symbol boyunca RE→UE atamasını çizer.
    • Boş RE yok; her RE’de mutlaka bir UE var.
    • Orijinal plasma colormap kullanılır.
    """
    # 1) 'time' kolonu: slot*num_symbols + sym
    alloc = alloc_df.copy()
    alloc['time'] = alloc.slot * num_symbols + alloc.sym

    # 2) Pivot: index=sc, columns=time, values=ue
    mat = alloc.pivot(index="sc", columns="time", values="ue")
    mat = mat.iloc[::-1]  # subcarrier 0 alta gelsin

    # 3) UE listesi ve kodlama: {ue_id: kod}
    ue_list = sorted(alloc['ue'].dropna().unique().astype(int).tolist())
    code_map = {float(u): i for i, u in enumerate(ue_list)}

    # 4) mat içindeki UE id'leri → kodlara dönüştür
    #    (float mat → replace ile mapping → int DataFrame)
    mat_idx = mat.replace(code_map).astype(int)

    # 5) plasma colormap, discrete
    n_colors = len(ue_list)
    cmap    = plt.get_cmap('plasma', n_colors)
    vmin, vmax = -0.5, n_colors - 0.5

    plt.figure(figsize=(12,6))
    im = plt.imshow(mat_idx, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # 6) renk çubuğu: her UE için bir renk
    ticks  = list(range(n_colors))
    labels = [f'UE {u}' for u in ue_list]
    cb = plt.colorbar(im, ticks=ticks)
    cb.ax.set_yticklabels(labels)

    plt.title(title)
    plt.xlabel('Time (slot × symbol)')
    plt.ylabel('Subcarrier')
    plt.tight_layout()

