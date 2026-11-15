import React, { useState } from 'react';

function AccidentReportForm({ setPredictionResult }) {
  const [jsonInput, setJsonInput] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [base64Image, setBase64Image] = useState('');

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setBase64Image(reader.result);
        setImageFile(file);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const dataToSend = JSON.parse(jsonInput);
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ siniestro: dataToSend, image: base64Image })
      });
      const result = await response.json();
      const dataToLoad = JSON.parse(result);
      setPredictionResult(dataToLoad);
    } catch (error) {
      console.error('Error al enviar el JSON:', error);
      alert('Error en el JSON o en la conexión, revisa la consola para más detalles.');
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <textarea
        value={jsonInput}
        onChange={(e) => setJsonInput(e.target.value)}
        placeholder="Ingrese el JSON aquí..."
        rows="8"
        style={{ width: '100%', padding: '10px' }}
      />
      <input type="file" accept="image/*" onChange={handleImageChange} style={{ padding: '10px' }} />
      <button type="submit" style={{ padding: '10px', backgroundColor: '#08f', color: '#fff', cursor: 'pointer' }}>
        Enviar
      </button>
      {base64Image && <img src={base64Image} alt="Vista previa" style={{ maxWidth: '100%', height: 'auto' }} />}
    </form>
  );
}

export default AccidentReportForm;
