import lsgdual_impl as lsgdual
import time
# import gdual_impl as gdual
# import matplotlib.pyplot as plt
import numpy as np

N = 5;
# q_vals = [10,20,40,60,80,100,125,150,175,200,250,300,400]
q_vals = [200]
PLT_TITLE = 'compose (exp)'

rt_lsgdbk  = np.zeros(len(q_vals))
rt_lsgdstd = np.zeros(len(q_vals))
rt_gdbk    = np.zeros(len(q_vals))
rt_gdstd   = np.zeros(len(q_vals))

for i_q in range(len(q_vals)):
    q = q_vals[i_q]
    # s = lsgdual.lsgdual_xdx(0.5, 10)
    F = lsgdual.exp(lsgdual.lsgdual_xdx(7, q))
    G = lsgdual.exp(lsgdual.lsgdual_xdx(5, q))

    start = time.time()
    for i in range(1,N):
        H = lsgdual.compose_brent_kung(F, G)
    end = time.time()
    # print("lsgd:bk  %f" % (end - start))
    # rt_lsgdbk.append(end - start)
    rt_lsgdbk[i_q] = end - start

    # start = time.time()
    # for i in range(1,N):
    #     H = lsgdual.compose(F, G)
    # end = time.time()
    # # print("lsgd:std %f" % (end - start))
    # # rt_lsgdstd.append(end - start)
    # rt_lsgdstd[i_q] = end - start
    #
    # F = lsgdual.lsgd2gd(F)
    # G = lsgdual.lsgd2gd(G)
    #
    # start = time.time()
    # for i in range(1,N):
    #     H = gdual.compose_brent_kung(F, G)
    # end = time.time()
    # # print("gd:bk    %f" % (end - start))
    # # rt_gdbk.append(end - start)
    # rt_gdbk[i_q] = end - start
    #
    # start = time.time()
    # for i in range(1,N):
    #     H = gdual.compose(F, G)
    # end = time.time()
    # # print("gd:std   %f" % (end - start))
    # # rt_gdstd.append(end - start)
    # rt_gdstd[i_q] = end - start

# plt.plot(q_vals, rt_lsgdbk, q_vals, rt_lsgdstd, q_vals, rt_gdbk, q_vals, rt_gdstd)
# plt.legend(('lsgd:bk', 'lsgd:std', 'gd:bk', 'gd:std'))
# plt.ylabel('rt (s)')
# plt.xlabel('q')
# plt.title(PLT_TITLE)
# plt.savefig(PLT_TITLE + '.png')
# plt.show()