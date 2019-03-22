
def spm_hrf(RT, T=16):
'''
% RT   - scan repeat time
% p    - parameters of the response function (two Gamma functions)
% T    - microtime resolution [Default: 16]
'''
    dt = RT / T
    u = np.arange(0, np.ceil(32)/dt)
    hrf = 
    hrf = hrf[np.arange(0,np.floor(32/RT)) * T]
    hrf /= np.sum(hrf)