
import matplotlib.pyplot as plt 
import numpy as np 
import json
import pandas as pd 
import corner
import scienceplots # noqa: F401
#from scipy import interpolate
import warnings
import random
#from parse import * 
warnings.filterwarnings("error")
plt.style.use('science')



def plot_synthetic_data(t,state_phi,state_f,phi_measured,psr_index=1,state_phi_pred=None,state_f_pred=None,phi_measured_pred=None):

    #Setup the figure
    h,w = 12,8
    rows = 3
    cols = 1
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=True)
    #Variables to plot
    tplot = t / (365*24*3600)
    tplot=np.arange(len(t))
    state_phi_i = state_phi[:,psr_index]
    state_f_i   = state_f[:,psr_index]
    phi_measured_i = phi_measured[:,psr_index]

    #Plot 'em 
    axes[0].plot(tplot,state_phi_i)
    print("first state phase value = ", state_phi_i[0])

    axes[1].plot(tplot,state_f_i)
    print("first state frequency value = ", state_f_i[0])
    axes[2].plot(tplot,phi_measured_i)

    #Plot the predictions too, if you have them
    if state_phi_pred is not None:
        axes[0].plot(tplot,state_phi_pred[:,psr_index])
        #print("first predicted state phase value = ", state_phi_pred[0,psr_index])

    if state_f_pred is not None:
        axes[1].plot(tplot,state_f_pred[:,psr_index])
        #print("first predicted state frequency value = ", state_f_pred[0,psr_index])

    if phi_measured_pred is not None:
        axes[2].plot(tplot,phi_measured_pred[:,psr_index])
        #print("first predicted measured phase value = ", phi_measured_pred[0,psr_index])


    # #Make it pretty

    fs=20
    axes[2].set_xlabel('t [years]', fontsize=fs)
    axes[0].set_ylabel(r'$\phi^*$ [rad]', fontsize=fs)
    axes[1].set_ylabel(r'$f^*$ [Hz]', fontsize=fs)
    axes[2].set_ylabel(r'$\phi^*_{\rm m}$ [rad]', fontsize=fs)

    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    plt.suptitle(f'Synthetic data for PSR index: {psr_index}',fontsize=fs)

    for ax in axes:    
        ax.xaxis.set_tick_params(labelsize=fs-4)
        ax.yaxis.set_tick_params(labelsize=fs-4)





def plot_nested_sampling_results(path,injection_parameters=None, ranges=None,labels=None, variables_to_plot=None):


        print(f"Loading results from {path}")
        # Load the json results file
        f = open(path)
        data = json.load(f)
        df_posterior = pd.DataFrame(data["posterior"]["content"]) # posterior
        evidence = data["log_evidence"]
        f.close()

        #Set-up defaults
        if injection_parameters is None:
            injection_parameters = injection_parameters = [5e-7,0.20,2.50,1.0,1.0,1.0,5e-15]
        #if ranges is None: #for now let the ranges be whatever
            #ranges=[(4.95e-7,5.05e-7),(-0.2,1.0),(2.0,3.0),(-0.2,np.pi/2),(0.5,1.5),(0.5,1.5),(0.5*5e-15,1.5*5e-15)]
        if labels is None:
            labels = [r'$\Omega$',r'$\Phi_0$',r'$\psi$',r'$\iota$', r'$\delta$',r'$\alpha$',r'$h_{0, \times 10}$']
        if variables_to_plot is None:
            variables_to_plot = ["omega_gw","phi0_gw","psi_gw","iota_gw","delta_gw","alpha_gw", "h"]

        #Create a numpy array of the variables you want to plot
        y_post = df_posterior[variables_to_plot].to_numpy()

    
        #Now plot it using corner.corner
        fs = 20
        fig = corner.corner(y_post, 
                            color='C0',
                            show_titles=True,
                            smooth=True,smooth1d=True,
                            truth_color='C2',
                            quantiles=[0.16, 0.84], #[0.16, 0.84]
                            truths = injection_parameters,
                            range=ranges,
                            labels = labels,
                            label_kwargs=dict(fontsize=fs))
        print(f"Model evidence is {evidence}")
        print(f"The number of samples is {len(df_posterior)}")
            

        #Pretty-ify
        for ax in fig.axes:

            if ax.lines: #is anything plotted on this axis?
            
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))

                ax.yaxis.set_tick_params(labelsize=fs-6)
                ax.xaxis.set_tick_params(labelsize=fs-6)


                ax.title.set_size(18)
       


#     if savefig is not None:
#         plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)
        





      

    #By default, plot canonical values




        

#     y_post = df_posterior[variables_to_plot].to_numpy()


#     return y_post,injection_parameters,ranges,model_evidence



# def plot_custom_corner(path,variables_to_plot,labels,injection_parameters,ranges,axes_scales,scalings=[1.0,1.0],savefig=None,logscale=False,title=None,smooth=True,smooth1d=True,fig=None):
#     #Extract the data as a numpy array
#     y_post,injection_parameters,ranges,model_evidence= _extract_posterior_results(path,variables_to_plot,injection_parameters,ranges,scalings=scalings)


#     #Log scale the axes if needed
#     if logscale:
#         y_post = np.log10(y_post)
#         injection_parameters = np.log10(injection_parameters)
#         ranges = np.log10(ranges)




    

# def _drop_braces(string_object):

#     string_object = string_object.replace('{', '')
#     string_object = string_object.replace('}', '')
#     return string_object

# def _extract_value_from_title(title_string):

#     template = '$\\{param_name}$ = ${value}_{lower}^{upper}$'

#     parsed_output = parse(template, title_string)

#     if parsed_output is None: #Handles h which is not a greek letter
#         template = '${param_name}$ = ${value}_{lower}^{upper}$'
#         parsed_output = parse(template, title_string)


#     return parsed_output['param_name'],_drop_braces(parsed_output['value']),_drop_braces(parsed_output['lower']),_drop_braces(parsed_output['upper'])

# #https://stackoverflow.com/questions/32923605/is-there-a-way-to-get-the-index-of-the-median-in-python-in-one-command
# def _argmedian(x):
#     return np.argpartition(x, len(x) // 2)[len(x) // 2]

# def stacked_corner(list_of_files,number_of_files_to_plot,variables_to_plot,labels,injection_parameters,ranges,axes_scales,scalings=[1.0,1.0],savefig=None,logscale=False,title=None,smooth=True,smooth1d=True,seed=1):

#     #Some arrays to hold the title value returned by corner.corner
#     num_params = len(variables_to_plot)
#     title_values = np.zeros((num_params,number_of_files_to_plot)) #an array with shape number of parameters x number of noise realisations 
#     title_upper = np.zeros((num_params,number_of_files_to_plot)) 
#     title_lower = np.zeros((num_params,number_of_files_to_plot))


#     #Select some files at random
#     fig= None 
#     random.seed(seed)
#     selected_files = random.sample(list_of_files,number_of_files_to_plot)
#     error_files = []
#     for i,f in enumerate(selected_files):
#         injection_parameters_idx = injection_parameters.copy()
#         ranges_idx = ranges.copy()

#         y_post,injection_parameters_idx,ranges_idx= _extract_posterior_results(f,variables_to_plot,injection_parameters_idx,ranges_idx,scalings=scalings)

#         errors = get_posterior_accuracy(y_post,injection_parameters_idx,labels)
#         error_files.extend([errors])
#         k = i 
#         if k ==2:
#             k = k+1 #convoluted way of skipping C2 color. Surely a better way exists


#         if logscale:
#             yplot = np.log10(y_post)
#             injection_parameters = np.log10(injection_parameters)
 
#         else:
#             yplot =y_post
        
#         nsamples = len(y_post)
#         fs = 20
   
#         fig = corner.corner(yplot, 
#                             color=f'C{k}',
#                             show_titles=True,
#                             smooth=smooth,smooth1d=smooth1d,
#                             truth_color='C2',
#                             quantiles=None, #[0.16, 0.84],
#                             truths =injection_parameters_idx ,
#                             range=ranges_idx,
#                             labels = labels,
#                             label_kwargs=dict(fontsize=fs),
#                             axes_scales = axes_scales,
#                             weights = np.ones(nsamples)/nsamples,
#                             plot_datapoints=True,fig=fig)


#         #Extract the axis titles 
#         kk = 0
#         for ax in fig.axes:
#             ax_title = ax.get_title()
            
#             if ax_title != '':

               
#                 param_name, value,lower_limit,upper_limit = _extract_value_from_title(ax_title) #Get the values that corner.corner sends to the ax title
#                 title_values[kk,i] = value
#                 title_lower[kk,i] = lower_limit
#                 title_upper[kk,i] = upper_limit


            
#                 kk += 1
                


  


#     #Pretty-ify
#     ax_count = 0
#     for ax in fig.axes:
        
#         if ax.lines: #is anything plotted on this axis?            
#             if len(ax.lines) == 18:
#                 ax_count += 1

#             #ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            
#             if ax_count == 5: #very hacky way to pop off overlapping ytick
#                 print("Setting y major locator")
#                 print("This is for axis:", ax)
#                 ax.yaxis.set_major_locator(plt.MaxNLocator(2))
#             else:
#                 ax.yaxis.set_major_locator(plt.MaxNLocator(3))

            
#             ax.xaxis.set_major_locator(plt.MaxNLocator(3))

#             ax.yaxis.set_tick_params(labelsize=fs-6)
#             ax.xaxis.set_tick_params(labelsize=fs-6)

#             #Get all lines
#             lines = ax.lines
            
            

#         ax.title.set_size(18)



#     #Get the indices of the median values from the list of medians 
#     idxs = [] #this is the index of the median for each parameter. 
#     for l in range(num_params):
#         idx = _argmedian(title_values[l,:])
#         idxs.extend([idx])



#     #Now use it to set the titles
#     kk = 0
#     for ax in fig.axes:
#         ax_title = ax.get_title()
#         if ax_title != '':


#             selected_idx = idxs[kk]



#             new_title_string = rf'{labels[kk]} $= {title_values[kk,selected_idx]:.2f}_{{{title_lower[kk,selected_idx]:.2f}}}^{{+{title_upper[kk,selected_idx]:.2f}}}$'
#             ax.set_title(new_title_string, fontsize=18)
#             kk += 1





#     if savefig != None:
#         plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)



#     #Surface some numbers
#     print("Surfacing some numbers for comparing two posteriors")
#     if len(selected_files) ==2:
#         errors1 = error_files[0]
#         errors2 = error_files[1]

#         #print(errors1)
#         #print(errors2)
#         #relative_error = (errors2 - errors1) / errors1
#         relative_error = (errors2 - errors1) #/ errors1

#         #print(relative_error)
#         for i in range(len(relative_error)):
#             print("%.3g" % errors1[i],"%.3g" %errors2[i],"%.3g" %relative_error[i]) #printing to 3 sig fig



# def get_posterior_accuracy(posterior,injection,labels):

#     print("The error in the 1D posteriors is as follows:")
#     rmse_errors =np.zeros(posterior.shape[-1])
#     for i in range(posterior.shape[-1]):
#         y = posterior[:,i]
#         inj = injection[i]
#         error = np.mean(np.abs(inj - y) / inj) #julian error

#         rmse = np.sqrt(np.sum((y - inj)**2) / len(y))
#         #rmse_errors[i] = rmse
#         rmse_errors[i] = error

#         print(labels[i], error,rmse)
#     print('*****************************')


#     return rmse_errors






# def plot_likelihood(x,y,parameter_name,log_x_axes=False,injection=1.0):

#     h,w = 8,8
#     rows = 1
#     cols = 1
#     fs =20
#     plt.style.use('science')
#     fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=False)


#     ax.plot(x,y)


#     ax.set_xlabel(parameter_name, fontsize=fs)
#     ax.set_ylabel(r'$\log \mathcal{L}$', fontsize=fs)
#     ax.yaxis.set_tick_params(labelsize=fs-6)
#     ax.xaxis.set_tick_params(labelsize=fs-6)

#     if log_x_axes:
#         ax.set_xscale('log')


#     ax.axvline(injection,c='0.5',linestyle='--')
#     plt.show()







# def SNR_plots(x,y1,y2,xlabel,savefig=None):

#     plt.style.use('science')
   
    

#     h,w = 12,8
#     rows = 1
#     cols = 1
#     fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(h,w),sharex=False)
    

#     ax.scatter(x,y1,label="Full PTA",c="C0")
#     ax.scatter(x,y2,label="Single Pulsar",c="C2")

    

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.axhline(7,linestyle='--', c='0.5')

#     f1 = interpolate.interp1d(y1, x)
#     xc = f1(7.0)
#     ax.axvline(xc,linestyle='--', c='C0')
#     print("Cutoff value y1 = ", xc)
#     idx = np.where(y1 > 7.0)[0]
#     ax.plot(x[idx],y1[idx],c="C0")
    
    
#     f2 = interpolate.interp1d(y2, x)
#     xc = f2(7.0)
#     ax.axvline(xc,linestyle='--', c='C2')
#     print("Cutoff value y2 = ", xc)
#     idx = np.where(y2 > 7.0)[0]
#     ax.plot(x[idx],y2[idx],c="C2")




#     fs=18
#     ax.set_xlabel(xlabel, fontsize=fs)
#     ax.set_ylabel(r'$\Lambda$', fontsize=fs)
    
#     ax.xaxis.set_tick_params(labelsize=fs-4)
#     ax.yaxis.set_tick_params(labelsize=fs-4)

#     ax.legend()
#     if savefig is not None:
#         plt.savefig(f"../data/images/{savefig}.png", bbox_inches="tight",dpi=300)













