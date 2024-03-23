// import ApiService from '@/services/api.service'

export interface DetectRepository {
    // detect(id: string | number): Promise<void>;
    detect(projectId: number, exampleId: number): Promise<void>;
    // public async clear(projectId: string, exampleId: number): Promise<void> {
    //     const url = this.baseUrl(projectId, exampleId)
    //     await this.request.delete(url)
    // };
};
// export abstract class DetectRepository<T> {
//     public async detect(projectId: string, exampleId: number): Promise<void> {
//         const url = this.baseUrl(projectId, exampleId)
//         await this.request.delete(url)
//     };
// }