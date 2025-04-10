# Imagen base con Miniconda
FROM public.ecr.aws/y0o4y9o3/anaconda-pkg-build:main

# Definir el directorio de trabajo
WORKDIR /app

# Copiar el archivo de dependencias
COPY environment.yml .

# Crear y activar el entorno Conda
RUN conda env create -f environment.yml && conda clean --all -y

# Activar el entorno por defecto al iniciar el contenedor
SHELL ["conda", "run", "-n", "mi_entorno", "/bin/bash", "-c"]

# Copiar el resto del código
COPY . .

# Definir el comando de ejecución
CMD ["conda", "run", "-n", "mi_entorno", "python", "app.py"]
