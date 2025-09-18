# fmt: off
import numpy as np
from statsforecast.ces import (
    FULL,
    NONE,
    PARTIAL,
    SIMPLE,
    auto_ces,
    cescalc,
    cesforecast,
    cesmodel,
    forecast_ces,
    forward_ces,
    initparamces,
    initstate,
    switch_ces,
)
from statsforecast.utils import AirPassengers as ap


class TestCES:
    @classmethod
    def setup_class(cls):
        """Set up class-level variables for all tests"""
        cls.m = 12
        cls.alpha_0_nonseas = 2.001457
        cls.alpha_1_nonseas = 1.000727
        cls.beta_0_nonseas = 0.
        cls.beta_1_nonseas = 0.
        
        cls.alpha_0_simple = 1.996411
        cls.alpha_1_simple = 1.206694
        cls.beta_0_simple = 0.
        cls.beta_1_simple = 0.
        
        cls.alpha_0_partial = 1.476837
        cls.alpha_1_partial = 1.
        cls.beta_0_partial = 0.91997
        cls.beta_1_partial = 0.
        
        cls.alpha_0_full = 1.350795
        cls.alpha_1_full = 1.009169
        cls.beta_0_full = 1.777909
        cls.beta_1_full = 0.973739
        
        # Initialize states for non-seasonal tests
        cls.init_states_non_seas = np.zeros((2 + len(ap), 2), dtype=np.float32)
        cls.init_states_non_seas[0] = initstate(ap, cls.m, 'N')
        
    def test_nonseasonal(self):
        """Test nonseasonal CES model"""
        _nmse = len(ap)
        amse_ = np.zeros(30)
        e_ = np.zeros(len(ap))
        
        cescalc(y=ap,
                states=self.init_states_non_seas, m=self.m, 
                season=NONE, alpha_0=self.alpha_0_nonseas, 
                alpha_1=self.alpha_1_nonseas, beta_0=self.beta_0_nonseas, 
                beta_1=self.beta_1_nonseas,
                e=e_, amse=amse_, nmse=3, 
                backfit=1)
        np.testing.assert_array_equal(
            self.init_states_non_seas[[0, -2, -1]],
            np.array([
                [  112.06887, 1301.9882 ],
                [  430.92154 , 2040.1951 ],
                [  432.40475, -1612.2461 ]
            ], dtype=np.float32)
        )
    
    def test_nonseasonal_forecast(self):
        """Test nonseasonal forecast"""
        h = 13
        fcsts = np.zeros(h, dtype=np.float32)
        cesforecast(states=self.init_states_non_seas, n=len(ap), m=self.m, 
                    season=NONE, 
                    f=fcsts, h=h, 
                    alpha_0=self.alpha_0_nonseas, alpha_1=self.alpha_1_nonseas, 
                    beta_0=self.beta_0_nonseas, beta_1=self.beta_1_nonseas)
        # taken from R using ces(AirPassengers, h=13)
        np.testing.assert_array_almost_equal(
            fcsts,
            np.array([
                430.9211, 432.4049, 431.2324, 432.7212, 431.5439,
                433.0376, 431.8556, 433.3543, 432.1675, 433.6712,
                432.4796, 433.9884, 432.7920
            ], dtype=np.float32), 
            decimal=2
        )
    
    def test_simple_seasonal(self):
        """Test simple seasonal CES model"""
        _nmse = len(ap)
        amse_ = np.zeros(30)
        _lik = 0.
        e_ = np.zeros(len(ap))
        
        init_states_s_seas = np.zeros((self.m * 2 + len(ap), 2), dtype=np.float32)
        init_states_s_seas[:self.m] = initstate(ap, self.m, 'S')
        
        cescalc(y=ap, 
                states=init_states_s_seas, m=self.m, 
                season=SIMPLE, alpha_0=self.alpha_0_simple, 
                alpha_1=self.alpha_1_simple, beta_0=self.beta_0_simple, 
                beta_1=self.beta_1_simple,
                e=e_, amse=amse_, nmse=3, backfit=1)
        np.testing.assert_array_equal(
            init_states_s_seas[[0, 11, 145, 143 + self.m]],
            np.array([
                [130.49458 ,  36.591137],
                [135.21922 , 121.62022 ],
                [423.57788 , 252.81241 ],
                [505.3621  ,  95.29781 ]
            ], dtype=np.float32)
        )
    
    def test_simple_seasonal_forecast(self):
        """Test simple seasonal forecast"""
        # Set up and process states first
        init_states_s_seas = np.zeros((self.m * 2 + len(ap), 2), dtype=np.float32)
        init_states_s_seas[:self.m] = initstate(ap, self.m, 'S')
        
        # Process the data first to get proper states
        amse_ = np.zeros(30)
        e_ = np.zeros(len(ap))
        cescalc(y=ap, 
                states=init_states_s_seas, m=self.m, 
                season=SIMPLE, alpha_0=self.alpha_0_simple, 
                alpha_1=self.alpha_1_simple, beta_0=self.beta_0_simple, 
                beta_1=self.beta_1_simple,
                e=e_, amse=amse_, nmse=3, backfit=1)
        
        # Now do the forecast
        h = 13
        fcsts = np.zeros(h, dtype=np.float32)
        cesforecast(states=init_states_s_seas, n=len(ap), m=self.m, 
                    season=SIMPLE, 
                    f=fcsts, h=h, 
                    alpha_0=self.alpha_0_simple, alpha_1=self.alpha_1_simple, 
                    beta_0=self.beta_0_simple, beta_1=self.beta_1_simple)
        # taken from R using ces(AirPassengers, h=13, seasonality = 'simple')
        np.testing.assert_array_almost_equal(
            fcsts,
            np.array([
                446.2768, 423.5779, 481.4365, 514.7730, 533.5008,
                589.0500, 688.2703, 674.5891, 580.9486, 516.0776,
                449.7246, 505.3621, 507.9884
            ], dtype=np.float32), 
            decimal=2
        )

    def test_partial_seasonal(self):
        """Test partial seasonal CES model"""
        _nmse = len(ap)
        amse_ = np.zeros(30)
        _lik = 0.
        e_ = np.zeros(len(ap))
        
        init_states_p_seas = np.zeros((self.m + len(ap), 3), dtype=np.float32)
        init_states_p_seas[:self.m] = initstate(ap, self.m, 'P')
        
        cescalc(y=ap, 
                states=init_states_p_seas, m=self.m, 
                season=2, alpha_0=self.alpha_0_partial, 
                alpha_1=self.alpha_1_partial, beta_0=self.beta_0_partial, 
                beta_1=self.beta_1_partial,
                e=e_, amse=amse_, nmse=3, backfit=1)
        np.testing.assert_array_equal(
            init_states_p_seas[[0, 11, 145, 143 + self.m]],
            np.array([
                [122.580666,  83.00358 ,  -9.710966],
                [122.580666,  78.11936 ,  -4.655848],
                [438.5037  , 300.70374 , -25.55726 ],
                [438.5037  , 296.92316 ,  -7.581563]
            ], dtype=np.float32)
        )

    def test_partial_seasonal_forecast(self):
        """Test partial seasonal forecast"""
        # Set up and process states first
        init_states_p_seas = np.zeros((self.m + len(ap), 3), dtype=np.float32)
        init_states_p_seas[:self.m] = initstate(ap, self.m, 'P')
        
        # Process the data first to get proper states
        amse_ = np.zeros(30)
        e_ = np.zeros(len(ap))
        cescalc(y=ap, 
                states=init_states_p_seas, m=self.m, 
                season=2, alpha_0=self.alpha_0_partial, 
                alpha_1=self.alpha_1_partial, beta_0=self.beta_0_partial, 
                beta_1=self.beta_1_partial,
                e=e_, amse=amse_, nmse=3, backfit=1)
        
        # Now do the forecast
        h = 13
        fcsts = np.zeros(h, dtype=np.float32)
        cesforecast(states=init_states_p_seas, n=len(ap), m=self.m, 
                    season=PARTIAL, 
                    f=fcsts, h=h, 
                    alpha_0=self.alpha_0_partial, alpha_1=self.alpha_1_partial, 
                    beta_0=self.beta_0_partial, beta_1=self.beta_1_partial)
        # taken from R using ces(AirPassengers, h=13, seasonality = 'partial')
        np.testing.assert_array_almost_equal(
            fcsts,
            np.array([
                437.6247, 412.9464, 445.5811, 498.5370, 493.0405, 550.7443, 
                629.2205, 607.1793, 512.3455, 462.1260, 383.4097, 430.9221, 437.6247
            ], dtype=np.float32), 
            decimal=2
        )

    def test_full_seasonal(self):
        """Test full seasonal CES model"""
        _nmse = len(ap)
        amse_ = np.zeros(30)
        _lik = 0.
        e_ = np.zeros(len(ap))
        
        init_states_f_seas = np.zeros((self.m * 2 + len(ap), 4), dtype=np.float32)
        init_states_f_seas[:self.m] = initstate(ap, self.m, 'F')
        
        cescalc(y=ap,
                states=init_states_f_seas, m=self.m, 
                season=3, alpha_0=self.alpha_0_full, 
                alpha_1=self.alpha_1_full, beta_0=self.beta_0_full, 
                beta_1=self.beta_1_full,
                e=e_, amse=amse_, nmse=3, backfit=1)
        np.testing.assert_array_equal(
            init_states_f_seas[[0, 11, 145, 143 + self.m]],
            np.array([
                [ 227.74284 ,  167.7603  ,  -94.299805,  -39.623283],
                [ 211.48921 ,  155.72342 ,  -91.62251 ,  -82.953064],
                [ 533.1726  ,  372.95758 , -139.31824 , -125.856834],
                [ 564.9041  ,  404.3251  , -130.9048  , -137.33    ]
            ], dtype=np.float32)
        )

    def test_full_seasonal_forecast(self):
        """Test full seasonal forecast"""
        # Set up and process states first
        init_states_f_seas = np.zeros((self.m * 2 + len(ap), 4), dtype=np.float32)
        init_states_f_seas[:self.m] = initstate(ap, self.m, 'F')
        
        # Process the data first to get proper states
        amse_ = np.zeros(30)
        e_ = np.zeros(len(ap))
        cescalc(y=ap,
                states=init_states_f_seas, m=self.m, 
                season=3, alpha_0=self.alpha_0_full, 
                alpha_1=self.alpha_1_full, beta_0=self.beta_0_full, 
                beta_1=self.beta_1_full,
                e=e_, amse=amse_, nmse=3, backfit=1)
        
        # Now do the forecast
        h = 13
        fcsts = np.zeros(h, dtype=np.float32)
        cesforecast(states=init_states_f_seas, n=len(ap), m=self.m, 
                    season=FULL, 
                    f=fcsts, h=h, 
                    alpha_0=self.alpha_0_full, alpha_1=self.alpha_1_full, 
                    beta_0=self.beta_0_full, beta_1=self.beta_1_full)
        # taken from R using ces(AirPassengers, h=13, seasonality = 'full')
        np.testing.assert_array_almost_equal(
            fcsts,
            np.array([
                450.9262, 429.2925, 465.4771, 510.1799, 517.9913, 578.5654,
                655.9219, 638.6218, 542.0985, 498.1064, 431.3293, 477.3273,
                501.3757
            ], dtype=np.float32), 
            decimal=2
        )

    def test_cesmodel_and_initparamces(self):
        """Test cesmodel and initparamces functions"""
        initparamces(alpha_0=np.nan, alpha_1=np.nan, 
                    beta_0=np.nan, beta_1=np.nan, 
                    seasontype='N')
        switch_ces('N')
        res = cesmodel(
            y=ap, m=self.m, seasontype='N',
            alpha_0=np.nan,
            alpha_1=np.nan,
            beta_0=np.nan, 
            beta_1=np.nan,
            nmse=3
        )
        _fcst = forecast_ces(res, 12)

    def test_auto_ces_functionality(self):
        """Test auto_ces functionality"""
        res = auto_ces(ap, m=self.m, model='F')
        _fcst = forecast_ces(res, 12)
        
        res = auto_ces(ap, m=self.m)
        np.testing.assert_array_equal(
            forecast_ces(forward_ces(res, ap), h=12)['mean'],
            forecast_ces(res, h=12)['mean']
        )
        
        # Test transfer
        _fcst_transfer = forecast_ces(forward_ces(res, np.log(ap)), h=12)
        res_transfer = forward_ces(res, np.log(ap))
        for key in res_transfer['par']:
            np.testing.assert_array_equal(res['par'][key], res_transfer['par'][key])
        
        # Less than two seasonal periods removes seasonal component
        res_short = auto_ces(np.arange(23, dtype=np.float64), m=self.m)
        assert res_short['seasontype'] == 'N'
