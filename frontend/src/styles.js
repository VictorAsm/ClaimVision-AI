const styles = {
  appContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    backgroundColor: '#282c34',
    minHeight: '100vh',
    color: 'white',
    padding: '20px',
  },
  mainTitle: {
    marginBottom: '20px',
    textAlign: 'center',
  },
  contentContainer: {
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    maxWidth: '1200px',
    margin: '0 auto',
  },
  resultContainer: {
    backgroundColor: '#444',
    borderRadius: '8px',
    padding: '20px',
    width: '45%',
  },
  title: {
    margin: '0 0 10px 0',
  },
  subtitle: {
    margin: '0 0 5px 0',
  },
  jsonContainer: {
    marginTop: '20px',
  },
  pre: {
    backgroundColor: '#222',
    padding: '10px',
    borderRadius: '5px',
    whiteSpace: 'pre-wrap',
    wordWrap: 'break-word',
  },
  image: {
    maxWidth: '100%',
    borderRadius: '8px',
  },
};

export default styles;
