// CI/CD for fraud-sequence-transformer, on a Jenkins running on the machine with the GPU.
pipeline {
    agent any

    parameters {
        booleanParam(name: 'RUN_PIPELINE', defaultValue: false, description: 'run the Prefect flow end to end (GPU, ~1 h)')
    }

    environment {
        PATH = "${env.HOME}/.local/bin:${env.HOME}/.local/opt/node/bin:${env.PATH}"
    }

    stages {
        stage('Install') {
            steps { sh 'uv sync --group mlops' }
        }
        stage('Lint and unit tests') {
            steps {
                sh 'uv run ruff check src scripts tests flows'
                sh 'mkdir -p reports && uv run --group mlops pytest -q --junitxml=reports/pytest.xml'
            }
            post {
                always { junit 'reports/pytest.xml' }
            }
        }
        stage('Web build') {
            steps {
                dir('web') { sh 'npm ci && npm run build' }
            }
        }
        stage('Pipeline (Prefect, GPU)') {
            when { expression { params.RUN_PIPELINE } }
            steps { sh 'uv run --group mlops flows/pipeline.py --split random' }
        }
        stage('Quality gate') {
            steps { sh 'uv run scripts/quality_gate.py' }
        }
    }
}
