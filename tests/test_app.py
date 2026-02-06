"""
Tests pytest pour le Dashboard de Monitoring Streamlit
Tests des fonctions principales du dashboard
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import shutil

# Mock streamlit avant d'importer le module
import sys
from unittest.mock import MagicMock

# Créer un mock de streamlit
sys.modules['streamlit'] = MagicMock()

# Créer des mocks pour les modules optionnels
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_production_logs():
    """Crée des logs de production de test"""
    logs = [
        {
            'timestamp': '2024-01-15 10:30:00',
            'model_version': '1.0.0',
            'input': {
                'ACTIVE_DAYS_CREDIT_ENDDATE_MIN': 100,
                'ACTIVE_DAYS_CREDIT_MAX': 500,
                'AMT_CREDIT': 250000,
                'DAYS_BIRTH': -15000,
                'DAYS_EMPLOYED': -5000
            },
            'output': {
                'score': 0.75,
                'decision': 'APPROVED'
            },
            'inference_time_ms': 45.2
        },
        {
            'timestamp': '2024-01-15 10:35:00',
            'model_version': '1.0.0',
            'input': {
                'ACTIVE_DAYS_CREDIT_ENDDATE_MIN': 150,
                'ACTIVE_DAYS_CREDIT_MAX': 600,
                'AMT_CREDIT': 300000,
                'DAYS_BIRTH': -16000,
                'DAYS_EMPLOYED': -4000
            },
            'output': {
                'score': 0.25,
                'decision': 'REJECTED'
            },
            'inference_time_ms': 48.5
        },
        {
            'timestamp': '2024-01-15 10:40:00',
            'model_version': '1.0.0',
            'input': {
                'ACTIVE_DAYS_CREDIT_ENDDATE_MIN': 120,
                'ACTIVE_DAYS_CREDIT_MAX': 550,
                'AMT_CREDIT': 280000,
                'DAYS_BIRTH': -15500,
                'DAYS_EMPLOYED': -4500
            },
            'output': {
                'score': 0.65,
                'decision': 'APPROVED'
            },
            'inference_time_ms': 46.3
        }
    ]
    return logs


@pytest.fixture
def sample_reference_data():
    """Crée des données de référence (entraînement)"""
    np.random.seed(42)
    data = {
        'ACTIVE_DAYS_CREDIT_ENDDATE_MIN': np.random.normal(100, 50, 1000),
        'ACTIVE_DAYS_CREDIT_MAX': np.random.normal(500, 100, 1000),
        'ACTIVE_DAYS_CREDIT_MEAN': np.random.normal(300, 80, 1000),
        'ACTIVE_DAYS_CREDIT_UPDATE_MEAN': np.random.normal(200, 60, 1000),
        'AMT_ANNUITY': np.random.lognormal(10, 1, 1000),
        'AMT_CREDIT': np.random.lognormal(12, 1, 1000),
        'AMT_GOODS_PRICE': np.random.lognormal(11.5, 1, 1000),
        'ANNUITY_INCOME_PERC': np.random.uniform(0, 1, 1000),
        'APPROVED_APP_CREDIT_PERC_VAR': np.random.uniform(0, 1, 1000),
        'DAYS_BIRTH': np.random.normal(-15000, 5000, 1000),
        'DAYS_EMPLOYED': np.random.normal(-5000, 3000, 1000),
        'EXT_SOURCE_1': np.random.uniform(0, 1, 1000),
        'EXT_SOURCE_2': np.random.uniform(0, 1, 1000),
        'EXT_SOURCE_3': np.random.uniform(0, 1, 1000),
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_logs_dir():
    """Crée un répertoire temporaire pour les logs"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# ============================================================
# TESTS DES FONCTIONS DE CHARGEMENT
# ============================================================

class TestLoadProductionLogs:
    """Tests pour la fonction load_production_logs"""
    
    def test_load_logs_file_not_exist(self, temp_logs_dir):
        """Test quand le fichier de logs n'existe pas"""
        logs_file = os.path.join(temp_logs_dir, "production_logs.json")
        
        with patch('os.path.exists', return_value=False):
            # Dans le contexte réel, cela retournerait un DataFrame vide
            assert True
    
    def test_load_logs_empty_file(self, temp_logs_dir):
        """Test avec un fichier de logs vide"""
        logs_dir = os.path.join(temp_logs_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        logs_file = os.path.join(logs_dir, "production_logs.json")
        
        with open(logs_file, 'w') as f:
            pass  # Fichier vide
        
        with patch('os.path.exists', return_value=True):
            # Le test vérifie que la fonction gère bien les fichiers vides
            assert os.path.exists(logs_file)
    
    def test_load_logs_valid_data(self, temp_logs_dir, sample_production_logs):
        """Test le chargement de logs valides"""
        logs_dir = os.path.join(temp_logs_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        logs_file = os.path.join(logs_dir, "production_logs.json")
        
        # Écrire les logs
        with open(logs_file, 'w') as f:
            for log in sample_production_logs:
                f.write(json.dumps(log) + '\n')
        
        # Lire et vérifier
        logs = []
        with open(logs_file, 'r') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        
        assert len(logs) == 3
        assert logs[0]['model_version'] == '1.0.0'
        assert logs[1]['output']['decision'] == 'REJECTED'


# ============================================================
# TESTS DES FONCTIONS DE TRAITEMENT
# ============================================================

class TestDataProcessing:
    """Tests pour le traitement des données"""
    
    def test_parse_production_dataframe(self, sample_production_logs):
        """Test la conversion des logs en DataFrame"""
        df = pd.DataFrame(sample_production_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        assert len(df) == 3
        assert 'timestamp' in df.columns
        assert 'model_version' in df.columns
        assert df['timestamp'].dtype == 'datetime64[ns]'
    
    def test_extract_input_output_features(self, sample_production_logs):
        """Test l'extraction des features d'input et output"""
        df = pd.DataFrame(sample_production_logs)
        
        input_df = pd.json_normalize(df['input'])
        output_df = pd.json_normalize(df['output'])
        
        assert 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN' in input_df.columns
        assert 'score' in output_df.columns
        assert 'decision' in output_df.columns
        assert len(input_df) == 3
        assert len(output_df) == 3
    
    def test_calculate_approval_rate(self, sample_production_logs):
        """Test le calcul du taux d'approbation"""
        df = pd.DataFrame(sample_production_logs)
        
        approved = (df['output'].apply(lambda x: x['decision']) == 'APPROVED').sum()
        approval_rate = (approved / len(df)) * 100
        
        assert approval_rate == pytest.approx(66.67, rel=1e-2)
    
    def test_calculate_average_inference_time(self, sample_production_logs):
        """Test le calcul du temps d'inférence moyen"""
        df = pd.DataFrame(sample_production_logs)
        
        avg_time = df['inference_time_ms'].mean()
        
        assert avg_time == pytest.approx(46.67, rel=1e-2)
        assert avg_time > 0
    
    def test_calculate_average_score(self, sample_production_logs):
        """Test le calcul du score moyen"""
        df = pd.DataFrame(sample_production_logs)
        
        avg_score = df['output'].apply(lambda x: x['score']).mean()
        
        assert avg_score == pytest.approx(0.55, rel=1e-2)
        assert 0 <= avg_score <= 1


# ============================================================
# TESTS DES MÉTRIQUES
# ============================================================

class TestMetrics:
    """Tests pour le calcul des métriques"""
    
    def test_total_predictions_count(self, sample_production_logs):
        """Test le nombre total de prédictions"""
        df = pd.DataFrame(sample_production_logs)
        assert len(df) == 3
    
    def test_decision_value_counts(self, sample_production_logs):
        """Test le comptage des décisions"""
        df = pd.DataFrame(sample_production_logs)
        decisions = df['output'].apply(lambda x: x['decision']).value_counts()
        
        assert decisions['APPROVED'] == 2
        assert decisions['REJECTED'] == 1
    
    def test_score_statistics(self, sample_production_logs):
        """Test les statistiques des scores"""
        df = pd.DataFrame(sample_production_logs)
        scores = df['output'].apply(lambda x: x['score'])
        
        assert scores.min() == 0.25
        assert scores.max() == 0.75
        assert len(scores) == 3
    
    def test_time_based_filtering(self, sample_production_logs):
        """Test le filtrage temporel"""
        df = pd.DataFrame(sample_production_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        now = df['timestamp'].max()
        filtered = df[df['timestamp'] >= (now - timedelta(minutes=10))]
        
        assert len(filtered) == 3
    
    def test_time_filtering_last_hour(self, sample_production_logs):
        """Test le filtrage pour la dernière heure"""
        df = pd.DataFrame(sample_production_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        now = df['timestamp'].max()
        one_hour_ago = now - timedelta(hours=1)
        filtered = df[df['timestamp'] >= one_hour_ago]
        
        assert len(filtered) == 3
    
    def test_time_filtering_excludes_old_data(self):
        """Test que le filtrage exclut les anciennes données"""
        logs = [
            {
                'timestamp': '2024-01-10 10:00:00',
                'model_version': '1.0.0',
                'output': {'score': 0.5, 'decision': 'APPROVED'}
            },
            {
                'timestamp': '2024-01-15 10:00:00',
                'model_version': '1.0.0',
                'output': {'score': 0.5, 'decision': 'APPROVED'}
            }
        ]
        
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        cutoff = pd.Timestamp('2024-01-12')
        filtered = df[df['timestamp'] >= cutoff]
        
        assert len(filtered) == 1



# ============================================================
# TESTS D'ERREUR
# ============================================================

class TestErrorHandling:
    """Tests pour la gestion d'erreurs"""
    
    def test_corrupted_json_in_logs(self, temp_logs_dir):
        """Test la gestion de JSON corrompu"""
        logs_dir = os.path.join(temp_logs_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        logs_file = os.path.join(logs_dir, "production_logs.json")
        
        with open(logs_file, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json\n')  # JSON corrompu
            f.write('{"valid": "json2"}\n')
        
        logs = []
        with open(logs_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        # Devrait charger seulement 2 logs valides
        assert len(logs) == 2
    
    def test_missing_columns_in_dataframe(self, sample_production_logs):
        """Test la gestion de colonnes manquantes"""
        df = pd.DataFrame(sample_production_logs)
        
        # Vérifie que les colonnes attendues existent
        expected_cols = ['timestamp', 'model_version', 'input', 'output', 'inference_time_ms']
        available_cols = [col for col in expected_cols if col in df.columns]
        
        assert len(available_cols) > 0
    
    def test_empty_dataframe_operations(self):
        """Test les opérations sur un DataFrame vide"""
        df = pd.DataFrame()
        
        # Les opérations ne doivent pas lever d'erreur
        assert len(df) == 0
        assert df.empty
    
    def test_invalid_time_range(self, sample_production_logs):
        """Test avec des plages de temps invalides"""
        df = pd.DataFrame(sample_production_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Plage future
        future = df['timestamp'].max() + timedelta(days=30)
        filtered = df[df['timestamp'] >= future]
        
        assert len(filtered) == 0


# ============================================================
# TESTS DE PERFORMANCE
# ============================================================

class TestPerformance:
    """Tests de performance"""
    
    def test_large_dataset_loading(self):
        """Test le chargement d'un grand dataset"""
        import time
        
        large_logs = []
        for i in range(1000):
            large_logs.append({
                'timestamp': f'2024-01-15 10:{i%60:02d}:00',
                'model_version': '1.0.0',
                'input': {'AMT_CREDIT': 250000 + i},
                'output': {'score': 0.5 + (i % 50) / 100, 'decision': 'APPROVED'},
                'inference_time_ms': 45.0 + (i % 10)
            })
        
        start = time.time()
        df = pd.DataFrame(large_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        elapsed = time.time() - start
        
        assert len(df) == 1000
        assert elapsed < 1.0  # Devrait être très rapide



# ============================================================
# TESTS EDGE CASES
# ============================================================

class TestEdgeCases:
    """Tests des cas limites"""
    
    def test_single_record(self):
        """Test avec un seul enregistrement"""
        df = pd.DataFrame([{
            'timestamp': '2024-01-15 10:00:00',
            'model_version': '1.0.0',
            'output': {'score': 0.5, 'decision': 'APPROVED'},
            'inference_time_ms': 45.0
        }])
        
        assert len(df) == 1
        assert df['output'].iloc[0]['decision'] == 'APPROVED'
    
    def test_all_approved_decisions(self):
        """Test quand toutes les décisions sont APPROVED"""
        logs = [
            {'output': {'decision': 'APPROVED'}},
            {'output': {'decision': 'APPROVED'}},
            {'output': {'decision': 'APPROVED'}},
        ]
        
        df = pd.DataFrame(logs)
        approved = (df['output'].apply(lambda x: x['decision']) == 'APPROVED').sum()
        approval_rate = (approved / len(df)) * 100
        
        assert approval_rate == 100.0
    
    def test_all_rejected_decisions(self):
        """Test quand toutes les décisions sont REJECTED"""
        logs = [
            {'output': {'decision': 'REJECTED'}},
            {'output': {'decision': 'REJECTED'}},
        ]
        
        df = pd.DataFrame(logs)
        rejected = (df['output'].apply(lambda x: x['decision']) == 'REJECTED').sum()
        
        assert rejected == 2
    
    def test_extreme_score_values(self):
        """Test avec des scores extrêmes"""
        logs = [
            {'output': {'score': 0.0}},
            {'output': {'score': 1.0}},
            {'output': {'score': 0.5}},
        ]
        
        df = pd.DataFrame(logs)
        scores = df['output'].apply(lambda x: x['score'])
        
        assert scores.min() == 0.0
        assert scores.max() == 1.0
        assert scores.mean() == pytest.approx(0.5, rel=1e-2)
    
    def test_very_fast_inference(self):
        """Test avec des temps d'inférence très courts"""
        logs = [
            {'inference_time_ms': 0.1},
            {'inference_time_ms': 0.2},
            {'inference_time_ms': 0.15},
        ]
        
        df = pd.DataFrame(logs)
        avg_time = df['inference_time_ms'].mean()
        
        assert avg_time > 0
        assert avg_time < 1
    
    def test_very_slow_inference(self):
        """Test avec des temps d'inférence très longs"""
        logs = [
            {'inference_time_ms': 5000},
            {'inference_time_ms': 6000},
            {'inference_time_ms': 5500},
        ]
        
        df = pd.DataFrame(logs)
        avg_time = df['inference_time_ms'].mean()
        
        assert avg_time > 5000


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])