import numpy as np
import nengo

### helper functions
def make_unitary_matrix_fourier( ssp_dim, domain_dim, eps=1e-3, rng = np.random, kernel = 'tophat'):
    if kernel == 'sinc':
        a = rng.rand( (ssp_dim - 1)//2, domain_dim )
        sign = rng.choice((-1, +1), size=np.shape(a) )
        phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    
    elif kernel == 'tophat':
        '''
        This doesn't work.
        '''
        fs = np.linspace( -100, 100, 1000000 )
        ps = np.abs(np.sinc(fs))
        ps /= ps.sum()
        phi = rng.choice( fs, p = ps, size = int((ssp_dim - 1)//2*domain_dim) ).reshape((ssp_dim - 1)//2, domain_dim)
        
    elif kernel == 'triangle':
        fs = np.linspace( -100, 100, 1000000 )
        ps = np.abs(np.sinc(fs))**2
        ps /= ps.sum()
        phi = rng.choice( fs, p = ps, size = int((ssp_dim - 1)//2*domain_dim) ).reshape((ssp_dim - 1)//2, domain_dim)

    fv = np.zeros( (ssp_dim,domain_dim), dtype='complex64')
    fv[0,:] = 1

    fv[1:(ssp_dim + 1) // 2,:] = phi
    fv[-1:ssp_dim // 2:-1,:] = -fv[1:(ssp_dim + 1) // 2,:]
    
    if ssp_dim % 2 == 0:
        fv[ssp_dim // 2,:] = 1

    return fv

class SSPEncoder:
    def __init__(self, phase_matrix, length_scale):
        '''
        Represents a domain using spatial semantic pointers.

        Parameters:
        -----------

        phase_matrix : cp.ndarray
            A ssp_dim x domain_dim ndarray representing the frequency 
            components of the SSP representation.

        length_scale : float or cp.ndarray
            Scales values before encoding.
        '''
        self.phase_matrix = phase_matrix
        self.domain_dim = self.phase_matrix.shape[1]
        self.ssp_dim = self.phase_matrix.shape[0]
        self.update_lengthscale(length_scale)

    def update_lengthscale(self, scale):
        '''
        Changes the lengthscale being used in the encoding.
        '''
        if not isinstance(scale, np.ndarray) or scale.size == 1:
            self.length_scale = scale * np.ones((self.domain_dim,))
        else:
            assert scale.size == self.domain_dim
            self.length_scale = scale
        assert self.length_scale.size == self.domain_dim
    
    def encode(self,x):
        '''
        Transforms input data into an SSP representation.

        Parameters:
        -----------
        x : cp.ndarray
            A (num_samples, domain_dim) array representing data to be encoded.

        Returns:
        --------
        data : cp.ndarray
            A (num_samples, ssp_dim) array of the ssp representation of the data
            
        '''
        
        x = np.atleast_2d(x)
        ls_mat = np.atleast_2d(np.diag(1/self.length_scale.flatten()))
        
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ scaled_x.T), axis=0 ).real
        
        return data.T   
    
def RandomSSPSpace(domain_dim, ssp_dim, length_scale = None, 
                   rng = np.random.default_rng(), 
                   kernel = 'tophat'):
    
    phase_matrix = make_unitary_matrix_fourier(ssp_dim,domain_dim, kernel = kernel )

    length_scale = np.array( length_scale )
    return SSPEncoder(phase_matrix, length_scale=length_scale)

n_pts = 100

walls = {}
walls['right'] = np.vstack( [np.ones((n_pts)), np.linspace(0.,1.,n_pts) ] ).T
walls['top'] = np.vstack( [np.linspace(0.,1.,n_pts),np.ones((n_pts)) ] ).T
walls['left'] = np.vstack( [np.zeros((n_pts)), np.linspace(0.,1.,n_pts) ] ).T
walls['bottom'] = np.vstack( [np.linspace(0.,1.,n_pts),np.zeros((n_pts)) ] ).T

# next, we will project the data for the walls into SSP Space
domain_dim = 2
encoder = RandomSSPSpace( domain_dim = domain_dim, ssp_dim = 4096, length_scale = 0.05, kernel = 'triangle' )
wall_phis = np.zeros( (4,encoder.ssp_dim) )
for w,(wall, data_xs) in enumerate(walls.items()):
    data_phis = encoder.encode( data_xs )
    wall_phi = data_phis.mean( axis = 0 )
    wall_phi /= np.linalg.norm( wall_phi )
    wall_phis[w,:] = wall_phi
encoders = wall_phis
    
def encode_ssp(t,  x ):
    phi = encoder.encode( x )
    return phi
    
n_neurons = wall_phis.shape[0]
    
model = nengo.Network()
with model:
    
    # user control
    user = nengo.Node( [0,0] )
    x = nengo.Node(size_in = domain_dim)
    nengo.Connection( user, x )
    
    
    # map position to ssp space
    phi = nengo.Node( encode_ssp, size_in = encoder.domain_dim, size_out = encoder.ssp_dim )
    nengo.Connection( x, phi, synapse = None)
    
    #encoders = nengo.dists.UniformHypersphere(surface=True).sample( n_neurons, encoder.ssp_dim )
    ens = nengo.Ensemble(
                    dimensions = encoder.ssp_dim,
                    n_neurons = n_neurons,
                    encoders = encoders,
                    neuron_type = nengo.SpikingRectifiedLinear(),
                    # max_rates = nengo.dists.Choice( [100] ),
                    bias = np.zeros( (n_neurons,)) - 10, 
                    gain = np.ones( (n_neurons,) ) * 200
                    )
                    
    nengo.Connection(phi,ens)
    