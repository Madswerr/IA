import http from 'k6/http';
import { sleep } from 'k6';
import { check, group } from 'k6';

// Configuración avanzada
export let options = {
    // Configuración de etapas escalonadas para simular crecimiento de usuarios
    stages: [
        { duration: '1m', target: 100 },  // 100 usuarios durante 1 minuto (pico de carga)
    ],
    thresholds: {
        http_req_duration: ['p(95)<2000'], // El 95% de las solicitudes debe tener una duración menor a 2000 ms
        'http_req_failed': ['rate<0.01'],  // Menos del 1% de las solicitudes deben fallar
    },
};

export default function() {
    group('Prueba de carga del sitio local de Vue', function () {
        // Apunta al servidor local
        const res = http.get('http://localhost:8080/');  // Cambia a la URL local

        // Validación de la respuesta HTTP
        check(res, {
            'status es 200': (r) => r.status === 200,
            'tiempo de respuesta menor a 2s': (r) => r.timings.duration < 2000,
        });
        
        // Simulación de tiempo de espera
        sleep(1);
    });
}
